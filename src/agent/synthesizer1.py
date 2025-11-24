from __future__ import annotations

import re
from datetime import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from config import get_settings
from openai import AzureOpenAI

from .reranker import ContextDoc, RerankResult

settings = get_settings()


@dataclass(slots=True)
class SynthesizedResponse:
    query: str
    answer: str
    contexts: List[ContextDoc]
    llm_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_sources(self) -> List[Dict[str, Any]]:
        """
        将上下文转换成便于展示的来源列表。
        """
        sources: List[Dict[str, Any]] = []
        for ctx in self.contexts:
            sources.append(
                {
                    "source": ctx.source,
                    "score": ctx.rerank_score,
                    "metadata": ctx.metadata,
                    "content": ctx.content,
                }
            )
        return sources


def _filter_contexts(
    contexts: Sequence[ContextDoc],
    max_k: int,
    max_chars: int,
) -> List[ContextDoc]:
    """
    根据排序结果做二次过滤：
    - 只取前 max_k 条
    - 同时控制拼接后的总字符数不超过 max_chars
    """
    if not contexts:
        return []

    selected: List[ContextDoc] = []
    used_chars = 0

    for doc in contexts[:max_k]:
        c = len(doc.content)
        if used_chars + c > max_chars:
            break
        selected.append(doc)
        used_chars += c

    return selected


class Synthesizer:
    def __init__(
        self,
        deployment_name: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        max_contexts: int = 8,
        max_context_chars: int = 12000,
    ) -> None:
        """
        :param deployment_name: Azure OpenAI 部署名
        :param api_key: 如不提供则从 Settings.azure_api_key 读取
        :param base_url: 如不提供则从 Settings.azure_url 读取
        :param system_prompt: 可选的系统提示词
        :param max_contexts: 最多使用多少段上下文
        :param max_context_chars: 上下文总字符数上限（防止 prompt 过长）
        """
        endpoint = base_url or settings.azure_url
        key = api_key or settings.azure_api_key
        if not endpoint or not key:
            raise RuntimeError(
                "Azure OpenAI endpoint / key not configured. "
                "请在环境变量 AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_KEY 中配置。"
            )

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version="2025-02-01-preview",
        )
        self.model = deployment_name
        self.max_contexts = max_contexts
        self.max_context_chars = max_context_chars

        if system_prompt is None:
            system_prompt = (
                "You are a retrieval-augmented assistant. "
                "You MUST carefully read the provided context snippets and "
                "answer the user's question using them as primary evidence. "
                "If the provided context contains relevant information "
                "(even if it is not perfectly up to date), use it to answer. "
                "If the context is insufficient, say that explicitly and "
                "briefly explain what is missing."
            )
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def synthesize(
        self,
        raw_query: str,
        query: str,
        rerank_result: RerankResult,
        top_k: Optional[int] = None,
    ) -> SynthesizedResponse:
        """
        根据 query 和 rerank 后的上下文，调用 LLM 生成最终回答。
        会自动识别 query 的语言（中/英），并用同样的语言回答。
        """
        if top_k is None:
            top_k = self.max_contexts

        # 1) 过滤上下文
        filtered_contexts = _filter_contexts(
            rerank_result.contexts,
            max_k=top_k,
            max_chars=self.max_context_chars,
        )

        # 2) 识别语言
        detected_language = self._detect_language(raw_query)

        # 3) 构造 messages
        messages = self._build_messages(
            query=query,
            contexts=filtered_contexts,
            detected_language=detected_language,
        )

        # 4) 调用 LLM
        start = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        elapsed = time.time() - start

        choice = resp.choices[0]
        answer_text = choice.message.content or ""

        usage_info = {}
        usage = getattr(resp, "usage", None)
        if usage is not None:
            usage_info = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        metadata: Dict[str, Any] = {
            "llm_model": resp.model,
            "llm_finish_reason": getattr(choice, "finish_reason", None),
            "llm_usage": usage_info,
            "llm_latency": elapsed,
        }

        return SynthesizedResponse(
            query=query,
            answer=answer_text,
            contexts=list(filtered_contexts),
            llm_metadata=metadata,
        )

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------
    def _detect_language(self, raw_query: str) -> str:
        """
        最简单的语言检测：看是否含有中文字符。
        返回 'zh' 或 'en'。
        """
        if not raw_query:
            return "en"

        # Unicode 范围：CJK Unified Ideographs
        pattern = re.compile('[\u4e00-\u9fff]')
        if pattern.search(raw_query):
            return "zh"
        return "en"

    def _build_messages(
        self,
        query: str,
        contexts: Sequence[ContextDoc],
        detected_language: str,
    ) -> List[Dict[str, str]]:
        """
        把系统提示词 + 上下文 + 用户问题拼成 ChatCompletion 所需的 messages。
        """
        # 构造上下文字符串
        context_lines: List[str] = []
        for idx, doc in enumerate(contexts, start=1):
            meta_str = ""
            source = doc.source or ""
            ts = doc.metadata.get("timestamp") if isinstance(doc.metadata, dict) else None
            if ts:
                meta_str = f"[source={source}, time={ts}]"
            elif source:
                meta_str = f"[source={source}]"

            context_lines.append(
                f"### Document {idx} {meta_str}\n{doc.content}"
            )

        context_block = "\n\n".join(context_lines) if context_lines else "No external context is available."

        # 语言指令
        if detected_language == "zh":
            lang_instruction = (
                "You MUST answer in **Chinese**. "
                "保持专业、简洁、条理清晰，可以分点作答。"
            )
        else:
            lang_instruction = (
                "You MUST answer in **English**. "
                "Be concise, well-structured, and professional."
            )

        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        user_content = (
            f"Current UTC time: {now_str}.\n"
            f"User question:\n{query}\n\n"
            f"Relevant context snippets:\n{context_block}\n\n"
            f"Instructions:\n"
            f"1. Base your answer primarily on the context snippets above.\n"
            f"2. If the context is insufficient or conflicting, say so explicitly.\n"
            f"3. Always mention the key sources you used (e.g. 'According to Document 1 ...').\n"
            f"4. Do NOT start with 'I cannot answer'. Go straight to the information you found.\n"
            f"{lang_instruction}"
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
