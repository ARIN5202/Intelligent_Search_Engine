#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重排序器 - 任务4
对检索结果进行精确重排序，筛选最相关内容。

使用流程：
1. 各个 retriever 返回 RetrievalResult。
2. 使用 gather_raw_contexts([...]) 合并为 ContextDoc 列表（跨来源）。
3. 用 Reranker.rerank(...) 或 rerank_from_results(...) 做精排序 + Top-k。
4. 将 RerankResult.contexts 作为上下文传给 LLM synthesizer。
"""

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
    从任意 RetrievalResult / RetrievedDocument 映射过来。
    """
    content: str
    source: str                               # 例如 "web_search", "local_rag", "finance"
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_score: Optional[float] = None  # 原始检索得分（可选）
    rerank_score: Optional[float] = None     # reranker 打分


@dataclass(slots=True)
class RerankResult:
    """rerank 之后给后续 Synthesizer 使用的结果。"""
    query: str
    contexts: List[ContextDoc]               # 已按 rerank_score 排好序的 top-k


def gather_raw_contexts(
    retrieval_results: Iterable[Any],  # 实际上是 Iterable[RetrievalResult]
) -> List[ContextDoc]:
    """
    将多个 RetrievalResult 展平成统一的 ContextDoc 列表。

    精确适配你的定义：

    RetrievalResult:
        - query: str
        - documents: List[RetrievedDocument]
        - provider: str         # retriever 名，例如 "finance_yf", "local_rag"
        - latency: Optional[float]
        - metadata: Dict[str, Any]

    RetrievedDocument:
        - content: str
        - source: str           # 文档来源标签，例如 "finance", "web"
        - score: float          # 检索得分
        - metadata: Dict[str, Any]

    映射策略：
      - ContextDoc.source:
            优先使用 doc.source，否则用 result.provider，否则 "unknown"
      - ContextDoc.metadata:
            先放入 result.metadata + {provider, latency}，
            再覆盖 doc.metadata（doc 的字段优先级更高）
      - ContextDoc.retrieval_score:
            使用 doc.score
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
            content = getattr(doc, "content", None)
            if not content:
                continue

            # doc metadata
            doc_meta_raw = getattr(doc, "metadata", None)
            doc_meta: Dict[str, Any] = {}
            if isinstance(doc_meta_raw, dict):
                doc_meta.update(doc_meta_raw)

            # 合并：doc 覆盖 result
            merged_meta = {**base_meta, **doc_meta}

            score = getattr(doc, "score", None)
            # doc.source 优先；否则回退到 provider；再否则 unknown
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


def _normalize_retrieval_scores(
        contexts: Sequence[ContextDoc],
) -> Dict[int, float]:
    """
    将 retrieval_score 线性归一化到 [0,1]。
    没有分数的记为 0；如果所有分数相同则统一为 1。
    """
    scores = [
        c.retrieval_score
        for c in contexts
        if c.retrieval_score is not None and not math.isnan(c.retrieval_score)
    ]
    if not scores:
        return {i: 0.0 for i in range(len(contexts))}

    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return {i: 1.0 for i in range(len(contexts))}

    norm: Dict[int, float] = {}
    for i, c in enumerate(contexts):
        s = c.retrieval_score
        if s is None or math.isnan(s):
            norm[i] = 0.0
        else:
            norm[i] = (s - lo) / (hi - lo)
    return norm


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps"
        self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.use_retrieval_score = use_retrieval_score
        self.alpha = alpha
        self.batch_size = batch_size

    @torch.no_grad()
    def _score_batch(
        self,
        query: str,
        docs: Sequence[ContextDoc],
    ) -> List[float]:
        """
        对一批 ContextDoc 用 cross-encoder 打分，返回与 docs 对齐的分数列表。
        """
        if not docs:
            return []

        queries = [query] * len(docs)
        passages = [d.content for d in docs]

        encoded = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        # 绝大多数 cross-encoder 是单输出：相关性分数
        logits = outputs.logits.squeeze(-1)  # [batch]
        return logits.detach().cpu().tolist()

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

        # 2) 融合检索得分（可选）
        if self.use_retrieval_score:
            norm_ret = _normalize_retrieval_scores(contexts)
            fused_scores: List[float] = []
            for i, ce in enumerate(ce_scores):
                fused = self.alpha * ce + (1.0 - self.alpha) * norm_ret[i]
                fused_scores.append(fused)
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
        retrieval_results: Iterable[Any],  # Iterable[RetrievalResult]
        top_k: int = 8,
    ) -> RerankResult:
        """
        方便用法：直接从多个 RetrievalResult 里做 gather + rerank。
        """
        contexts = gather_raw_contexts(retrieval_results)
        return self.rerank(query=query, contexts=contexts, top_k=top_k)
