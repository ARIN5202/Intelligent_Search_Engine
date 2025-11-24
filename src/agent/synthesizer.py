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

import re
from datetime import datetime
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from config import get_settings
from openai import AzureOpenAI

from .reranker import ContextDoc, RerankResult

settings = get_settings()  # 获取配置实例

@dataclass(slots=True)
class SynthesizedResponse:
    query: str
    answer: str
    contexts: List[ContextDoc]
    latency: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_sources(self) -> List[Dict[str, Any]]:
        """把 contexts 转成前端/CLI 用的 sources 列表."""
        sources = []
        for ctx in self.contexts:
            score = ctx.rerank_score
            if score is None:
                score = ctx.retrieval_score
            sources.append(
                {
                    "title": ctx.source,          # CLI 用的是 source['title']
                    "score": score if score is not None else 0.0,
                    "content": ctx.content,       # 你想展示的话也可以后面用
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
        c = doc.content
        length = len(c)
        if used_chars + length > max_chars:
            break
        selected.append(doc)
        used_chars += length
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
        Initialize the Synthesizer with required parameters.
        """
        endpoint = base_url or settings.azure_url
        key = api_key or settings.azure_api_key
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version="2025-02-01-preview",
        )
        self.model = deployment_name
        self.max_contexts = max_contexts
        self.max_context_chars = max_context_chars

        if system_prompt is None:
            system_prompt = ( """
                You are a knowledgeable, confident, and helpful AI assistant. 
                Your task is to synthesize a clear, direct, and accurate answer to the user's query based on the provided context. 
                Key Guidelines:
                1. USE THE CONTEXT: Prioritize the provided context information.
                2. FILL GAPS GRACEFULLY: If the context is partial or slightly outdated, use it to form the best possible answer. 
                   If the context is completely missing for a specific detail, rely on your general knowledge to provide a well-rounded answer without mentioning the absence of context.
                3. TONE: Be professional and direct. Do not begin responses with apologies (e.g., 'Sorry', 'I cannot'). Avoid phrases like 'The documents do not mention', 'There is insufficient evidence', or 'According to the limited context'. Simply state the facts.
                4. EMERGENCY/ALERTS: If a query involves potential disaster or emergency warnings, respond with reassurance: "No issues detected" or a similar calm, clear response.
                """
            )
        self.system_prompt = system_prompt

    def synthesize(
        self,
        raw_query: str,
        query: str,
        rerank_result: RerankResult,
        top_k: Optional[int] = None,
    ) -> SynthesizedResponse:
        """
        Synthesize the final response based on query and contexts.
        Now includes language detection to determine the response language.
        """
        if top_k is None:
            top_k = self.max_contexts

        # 1) Filter contexts (Top-k + Length restriction)
        filtered_contexts = _filter_contexts(
            rerank_result.contexts,
            max_k=top_k,
            max_chars=self.max_context_chars,
        )

        # 2) Detect the language of the raw query
        detected_language = self._detect_language(raw_query)

        # 3) Build messages with instructions for the language
        messages = self._build_messages(query, filtered_contexts, detected_language)

        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
        )
        elapsed = time.perf_counter() - start

        # 4) Process the response
        choice = resp.choices[0]
        answer_text = choice.message.content or ""

        # Collect usage information
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
            answer=answer_text.strip(),
            contexts=filtered_contexts,
            latency=elapsed,
            metadata=metadata,
        )

    # ---------- Internal helpers ----------

    def _detect_language(self, raw_query: str) -> str:
        """
        Detect the language of the raw_query using Regex.
        Returns "zh" if CJK characters are found, otherwise "en".
        """
        if not raw_query:
            return "en"

        # Unicode range for CJK Unified Ideographs (covers most common Chinese characters)
        # \u4e00-\u9fff is the standard range for CJK characters
        pattern = re.compile(r'[\u4e00-\u9fff]')

        if pattern.search(raw_query):
            return "zh"
        else:
            return "en"

    def _build_messages(
            self,
            query: str,
            contexts: Sequence[ContextDoc],
            detected_language: str,
    ) -> List[Dict[str, str]]:
        """
        Build messages with XML structure and clearer instructions.
        """
        now = datetime.now()
        current_time_str = now.strftime("%Y-%m-%d (%A) %H:%M")

        # 2. 构建 Context Block (使用 XML 结构)
        if contexts:
            doc_lines = []
            for i, c in enumerate(contexts, start=1):
                # 添加元数据，帮助 AI 判断信息的新旧和质量
                meta_info = f'source="{c.source}"'
                if c.rerank_score is not None:
                    meta_info += f' score="{c.rerank_score:.4f}"'

                # 使用 xml 标签包裹每个文档
                doc_lines.append(
                    f'<document index="{i}" {meta_info}>\n{c.content.strip()}\n</document>'
                )
            ctx_block = "\n".join(doc_lines)
            # 外层再包一个 context 标签
            full_context = f"<context>\n{ctx_block}\n</context>"
        else:
            full_context = "<context>\nNo external documents retrieved.\n</context>"

        # 3. 根据语言调整指令 (更加具体的语气)
        if detected_language == "zh":
            lang_instruction = (
                "請使用流暢、專業的繁體中文回答。\n"
                "如果文件中沒有直接答案，請根據你的常識對該主題進行一般性說明。\n"
                "**重要：用戶提到的'今天'、'週三'等相對時間，必須根據當前日期進行換算。**"
            )
        else:
            lang_instruction = (
                "Please answer in fluent, professional English.\n"
                "If the specific answer is not in the docs, provide a general explanation.\n"
                "**Important: Resolve relative time terms (e.g., 'today', 'Wednesday') based on Current Date.**"
            )

        # 4. 构建最终 User Content
        # 注意：将 Query 放在 Context 之后通常效果更好，符合阅读逻辑
        user_content = (
            f"Current Date: {current_time_str}\n\n"
            f"{full_context}\n\n"
            f"User Query: {query.strip()}\n\n"
            f"Instructions:\n"
            f"1. Analyze the <context> to answer the Query.\n"
            f"2. **Time Awareness**: You must interpret relative time references (like 'this Wednesday', 'last week') based strictly on the 'Current Date' provided above.\n"
            f"3. If context is insufficient, answer generally based on knowledge without complaining about missing data.\n"
            f"4. Only output plain text, do not use markdown.\n"
            f"{lang_instruction}"
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
