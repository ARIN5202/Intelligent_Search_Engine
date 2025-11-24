from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass(slots=True)
class ContextDoc:
    """
    Reranker 使用的统一上下文表示。
    会从各个检索器返回的 RetrievedDocument 映射到这个结构。
    """
    content: str
    source: str                               # 例如 "web_search", "local_rag", "finance"
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_score: Optional[float] = None  # 原始检索得分（可选）
    rerank_score: Optional[float] = None     # reranker 打分（稍后填充）


@dataclass(slots=True)
class RerankResult:
    """rerank 之后给后续 Synthesizer 使用的结果。"""
    query: str
    contexts: List[ContextDoc]               # 已按 rerank_score 排好序的 top-k


def gather_raw_contexts(
    retrieval_results: Iterable[Any],
) -> List[ContextDoc]:
    """
    将多个 RetrievalResult 展平成统一的 ContextDoc 列表。

    这里假设每个 RetrievalResult 至少包含：
        - documents: List[RetrievedDocument]
        - provider: str           # retriever 名，例如 "finance_yf", "local_rag"
        - latency: Optional[float]
        - metadata: Dict[str, Any] （可选）

    而每个 RetrievedDocument 至少包含：
        - content: str
        - source: str             # 文档来源标签，例如 "finance", "web"
        - score: float            # 检索得分
        - metadata: Dict[str, Any] （可选）
    """
    contexts: List[ContextDoc] = []

    for res in retrieval_results:
        if res is None:
            continue
        if not hasattr(res, "documents"):
            continue

        provider = getattr(res, "provider", None)
        latency = getattr(res, "latency", None)

        # result 级别元数据
        base_meta: Dict[str, Any] = {}
        res_meta = getattr(res, "metadata", None)
        if isinstance(res_meta, dict):
            base_meta.update(res_meta)

        if provider is not None:
            base_meta.setdefault("provider", provider)
        if latency is not None:
            base_meta.setdefault("latency", latency)

        documents = getattr(res, "documents", []) or []
        for doc in documents:
            if doc is None:
                continue

            # 支持多种字段名：content / page_content / text
            content: Optional[str] = getattr(doc, "content", None)
            if not content:
                content = getattr(doc, "page_content", None)
            if not content:
                content = getattr(doc, "text", None)
            if not content:
                # 如果还是拿不到，就跳过
                continue

            doc_meta = getattr(doc, "metadata", None)
            if isinstance(doc_meta, dict):
                merged_meta = {**base_meta, **doc_meta}
            else:
                merged_meta = dict(base_meta)

            score = getattr(doc, "score", None)
            # doc.source 优先；否则回退到 provider；再否则 "unknown"
            doc_source = getattr(doc, "source", None) or provider or "unknown"

            contexts.append(
                ContextDoc(
                    content=content,
                    source=doc_source,
                    metadata=merged_meta,
                    retrieval_score=score,
                )
            )

    return contexts


class Reranker:
    """
    重排序器：
    - 使用 Cross-Encoder 对 (query, context) 做精确打分。
    - 可选融合原始 retrieval_score。
    - 输出按相关性排序后的 top-k ContextDoc。
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_retrieval_score: bool = True,
        alpha: float = 0.7,
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> None:
        """
        :param model_name: HuggingFace cross-encoder 模型名
        :param use_retrieval_score: 是否融合原始检索得分
        :param alpha: 融合权重 in [0,1]:
                      final = alpha * cross_score + (1-alpha) * norm(retrieval_score)
        :param batch_size: 推理 batch 大小
        :param device: "cuda" / "cpu"，默认自动选择
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.use_retrieval_score = use_retrieval_score
        self.alpha = float(alpha)
        self.batch_size = int(batch_size)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _score_batch(self, query: str, batch_docs: Sequence[ContextDoc]) -> List[float]:
        """
        对一个 batch 的 (query, doc.content) 进行打分。
        返回 cross-encoder 的分数列表（越大相关性越高）。
        """
        pairs = [(query, doc.content) for doc in batch_docs]
        encoded = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            # 部分 cross-encoder 是 regression 任务，logits shape: [batch, 1]
            logits = outputs.logits.squeeze(-1)
            scores = logits.detach().cpu().tolist()

        # 保证返回 Python float list
        if isinstance(scores, float):
            scores = [float(scores)]
        else:
            scores = [float(s) for s in scores]
        return scores

    # ------------------------------------------------------------------
    # public APIs
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        contexts: Sequence[ContextDoc],
        top_k: int = 8,
    ) -> RerankResult:
        """
        对已有 ContextDoc 列表进行重排序并返回 top_k。
        """
        if not contexts:
            return RerankResult(query=query, contexts=[])

        # 1) Cross-Encoder 打分（分批跑）
        ce_scores: List[float] = []
        for i in range(0, len(contexts), self.batch_size):
            batch = contexts[i : i + self.batch_size]
            ce_scores.extend(self._score_batch(query, batch))

        # 2) 如需融合原始检索得分，则先做归一化，再 convex combination
        if self.use_retrieval_score:
            ret_scores = [doc.retrieval_score for doc in contexts]
            # 如果全部是 None，就不融合
            if any(s is not None for s in ret_scores):
                # 将 None 替换为最小值
                valid_scores = [s for s in ret_scores if s is not None]
                min_s = min(valid_scores)
                max_s = max(valid_scores)
                if math.isclose(max_s, min_s):
                    norm_ret = [0.0 for _ in ret_scores]
                else:
                    norm_ret = [
                        ((s if s is not None else min_s) - min_s) / (max_s - min_s)
                        for s in ret_scores
                    ]
                fused_scores = [
                    self.alpha * ce + (1.0 - self.alpha) * nr
                    for ce, nr in zip(ce_scores, norm_ret)
                ]
            else:
                fused_scores = ce_scores
        else:
            fused_scores = ce_scores

        # 3) 写回 ContextDoc，按得分排序取 top_k
        for doc, s in zip(contexts, fused_scores):
            doc.rerank_score = float(s)

        sorted_ctxs = sorted(
            contexts,
            key=lambda d: d.rerank_score if d.rerank_score is not None else -1e18,
            reverse=True,
        )

        return RerankResult(
            query=query,
            contexts=sorted_ctxs[:top_k],
        )

    def rerank_from_results(
        self,
        query: str,
        retrieval_results: Iterable[Any],
        top_k: int = 8,
    ) -> RerankResult:
        """
        方便用法：直接从多个 RetrievalResult 里做 gather + rerank。
        """
        contexts = gather_raw_contexts(retrieval_results)
        return self.rerank(query=query, contexts=contexts, top_k=top_k)
