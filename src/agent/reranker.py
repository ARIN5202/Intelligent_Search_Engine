#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重排序器 - 任务4
对检索结果进行精确重排序，筛选最相关内容
"""

import asyncio
from typing import List, Dict, Any, Tuple
import math


class Reranker:
    """重排序器，使用更精确的模型对检索结果进行重排序"""

    def __init__(self):
        """初始化重排序器"""
        # TODO: 在实际项目中这里应该加载Cross-Encoder模型
        # 例如: sentence-transformers/ms-marco-MiniLM-L-6-v2
        pass

    async def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        对文档进行重排序

        Args:
            query: 用户查询
            documents: 待排序的文档列表
            top_k: 返回的文档数量

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        # 计算相关性分数
        scored_docs = []
        for doc in documents:
            score = await self._calculate_relevance_score(query, doc)
            scored_docs.append((score, doc))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 返回top_k个结果
        reranked_docs = []
        for i, (score, doc) in enumerate(scored_docs[:top_k]):
            doc_with_score = doc.copy()
            doc_with_score['rerank_score'] = score
            doc_with_score['rerank_position'] = i + 1
            reranked_docs.append(doc_with_score)

        return reranked_docs

    async def _calculate_relevance_score(self, query: str, document: Dict[str, Any]) -> float:
        """
        计算查询和文档的相关性分数

        Args:
            query: 用户查询
            document: 文档

        Returns:
            相关性分数 (0-1之间)
        """
        # 简单的基于关键词重叠的相关性计算
        # 在实际项目中应该使用预训练的Cross-Encoder模型

        doc_text = document.get('content', '') + ' ' + document.get('title', '')
        doc_text = doc_text.lower()
        query_words = set(query.lower().split())
        doc_words = set(doc_text.split())

        # 计算词汇重叠
        intersection = query_words.intersection(doc_words)
        union = query_words.union(doc_words)

        jaccard_score = len(intersection) / len(union) if union else 0

        # 考虑文档长度和原始分数
        original_score = document.get('score', 0)
        doc_length_penalty = min(1.0, len(doc_text.split()) / 100)  # 适度惩罚过长文档

        # 综合分数
        final_score = (jaccard_score * 0.4 + original_score * 0.4 + doc_length_penalty * 0.2)

        return min(1.0, final_score)

    async def batch_rerank(self, queries: List[str], documents_list: List[List[Dict[str, Any]]], top_k: int = 5) -> \
    List[List[Dict[str, Any]]]:
        """批量重排序"""
        tasks = []
        for query, docs in zip(queries, documents_list):
            task = self.rerank(query, docs, top_k)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def health_check(self) -> bool:
        """健康检查"""
        return True