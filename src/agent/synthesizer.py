#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthesizer - Step 5-6
负责：
- 接收 RerankResult（已按相关性排好）
- 做上下文过滤（Top-k / 长度控制）
- 调用 DeepSeek API 做最终回答
- 记录延迟和 Token 使用情况
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from config import Settings, get_settings
from openai import OpenAI  # pip install openai

from .reranker import ContextDoc, RerankResult

settings = get_settings()  # 获取配置实例

@dataclass(slots=True)
class SynthesizedResponse:
    """
    LLM 综合后的最终结果。
    """
    query: str
    answer: str
    contexts: List[ContextDoc]          # 实际用于生成回答的上下文（已过滤）
    latency: float                     # LLM 调用耗时（秒）
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        c = doc.content
        length = len(c)
        if used_chars + length > max_chars:
            break
        selected.append(doc)
        used_chars += length

    return selected


class Synthesizer:
    """
    用法：
        synthesizer = Synthesizer(api_key=..., model="deepseek-chat")
        final = synthesizer.synthesize(query, rerank_result, top_k=8)

    上层就拿 SynthesizedResponse.answer 给用户即可。
    """

    def __init__(
        self,
        api_key: str,
        model: str = settings.deepseek_model,
        base_url: str = settings.deepseek_url,
        system_prompt: str | None = None,
        max_contexts: int = 8,
        max_context_chars: int = 12000,
    ) -> None:
        """
        :param api_key: DeepSeek API Key
        :param model:   DeepSeek 模型名，如 "deepseek-chat" / "deepseek-reasoner"
        :param base_url: DeepSeek API base url
        :param system_prompt: 系统提示词，用于规范回答风格和使用上下文规则
        :param max_contexts: 最多使用多少条 context（再上游已经是 Top-k，这里是保险）
        :param max_context_chars: 拼接后的 context 字符总长上限，避免 prompt 过长
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_contexts = max_contexts
        self.max_context_chars = max_context_chars

        if system_prompt is None:
            system_prompt = (
                "You are a retrieval-augmented assistant. "
                "You MUST carefully read the provided context snippets and "
                "answer the user's question using them as primary evidence. "
                "If the context is insufficient or conflicting, say so clearly "
                "and reason cautiously."
            )
        self.system_prompt = system_prompt


    def synthesize(
        self,
        query: str,
        rerank_result: RerankResult,
        top_k: Optional[int] = None,
    ) -> SynthesizedResponse:
        """
        主入口：
        - 从 RerankResult 中拿到排好序的 contexts
        - 再做一次 Top-k / 长度过滤
        - 调用 DeepSeek 生成最终回答
        - 返回带 latency / usage 的结构化结果
        """
        if top_k is None:
            top_k = self.max_contexts

        # 1) 上下文过滤（Top-k + 长度限制）
        filtered_contexts = _filter_contexts(
            rerank_result.contexts,
            max_k=top_k,
            max_chars=self.max_context_chars,
        )

        # 2) 构造 messages
        messages = self._build_messages(query, filtered_contexts)
        # 3) 调用 DeepSeek + 记录耗时
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
        )
        elapsed = time.perf_counter() - start

        # 4) 解析回答和 usage
        choice = resp.choices[0]
        answer_text = choice.message.content or ""

        usage_info = {}
        usage = getattr(resp, "usage", None)
        if usage is not None:
            # openai SDK 返回的是对象，这里稳妥一点 getattr
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
            answer=answer_text.strip(),
            contexts=filtered_contexts,
            latency=elapsed,
            metadata=metadata,
        )

    # ---------- Internal helpers ----------

    def _build_messages(
        self,
        query: str,
        contexts: Sequence[ContextDoc],
    ) -> List[Dict[str, str]]:
        """
        构造发给 DeepSeek 的 messages（OpenAI 风格）。
        把所有 context 组合成一个用户侧提示的一部分。
        """

        # 把 context 格式化清晰，方便模型引用
        if contexts:
            ctx_lines: List[str] = []
            for i, c in enumerate(contexts, start=1):
                src = c.source
                score = (
                    f" | rerank_score={c.rerank_score:.4f}"
                    if c.rerank_score is not None
                    else ""
                )
                ctx_lines.append(
                    f"[{i}] (source={src}{score})\n{c.content.strip()}"
                )
            ctx_block = "\n\n".join(ctx_lines)
        else:
            ctx_block = "No external contexts were retrieved."

        user_content = (
            f"User Question:\n{query.strip()}\n\n"
            f"Retrieved Contexts (use them as primary evidence when relevant):\n"
            f"{ctx_block}\n\n"
            "Instructions:\n"
            "- First, infer the answer based on the contexts above.\n"
            "- If multiple snippets conflict, explain the conflict and choose the most reliable.\n"
            "- If key information is missing, explicitly say what is missing instead of fabricating.\n"
            "- Answer in a clear, concise, and well-structured way."
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
