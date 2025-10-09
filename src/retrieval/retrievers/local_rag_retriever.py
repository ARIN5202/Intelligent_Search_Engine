#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地RAG检索器
从本地知识库中检索相关信息
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any
from .base_retriever import BaseRetriever


class LocalRAGRetriever(BaseRetriever):
    """本地RAG检索器"""

    def __init__(self):
        super().__init__("LocalRAG")
        self.description = "本地知识库检索器，搜索公司内部文档和规章制度"
        self.index_path = Path(__file__).parent.parent.parent.parent / "storage"
        self.documents = []  # 简单存储，实际应该使用向量数据库

    async def _do_initialize(self):
        """初始化本地索引"""
        # TODO: 在实际项目中这里应该加载向量索引
        # 例如: FAISS, ChromaDB, 或其他向量数据库

        # 模拟加载本地文档
        await self._load_local_documents()
        print(f"本地RAG检索器初始化完成，加载了 {len(self.documents)} 个文档")

    async def _load_local_documents(self):
        """加载本地文档（模拟实现）"""
        # 这里是模拟数据，实际应该从存储的向量索引中加载
        self.documents = [
            {
                'content': '员工应准时上班，迟到超过30分钟视为旷工。请假需要提前一天申请，紧急情况除外。',
                'title': '考勤制度',
                'source': 'company_rules.txt',
                'type': 'policy'
            },
            {
                'content': '报销需要提供正规发票。超过500元的支出需要经理审批。月度预算不得超支20%。',
                'title': '财务制度',
                'source': 'company_rules.txt',
                'type': 'finance'
            },
            {
                'content': '保持工作环境整洁。禁止在办公区域吸烟。下班前需要整理好个人工作台。',
                'title': '工作规范',
                'source': 'company_rules.txt',
                'type': 'workplace'
            }
        ]

    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        从本地知识库检索相关信息

        Args:
            query: 查询内容
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        if not self.is_initialized:
            await self.initialize()

        # 提取查询关键词
        keywords = self._extract_keywords(query)

        # 计算相关性分数
        scored_docs = []
        for doc in self.documents:
            score = self._calculate_similarity(query, keywords, doc)
            if score > 0:
                scored_docs.append((score, doc))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 格式化结果
        results = []
        for score, doc in scored_docs[:top_k]:
            result = self._format_result(
                content=doc['content'],
                title=doc['title'],
                source=doc['source'],
                score=score,
                metadata={
                    'type': doc.get('type', 'unknown'),
                    'retrieval_method': 'local_rag'
                }
            )
            results.append(result)

        return results

    def _calculate_similarity(self, query: str, keywords: List[str], document: Dict[str, Any]) -> float:
        """
        计算查询与文档的相似度

        Args:
            query: 原始查询
            keywords: 提取的关键词
            document: 文档

        Returns:
            相似度分数 (0-1)
        """
        doc_text = (document['content'] + ' ' + document['title']).lower()
        query_lower = query.lower()

        # 精确匹配得分
        exact_match_score = 0
        if query_lower in doc_text:
            exact_match_score = 0.5

        # 关键词匹配得分
        keyword_matches = 0
        for keyword in keywords:
            if keyword in doc_text:
                keyword_matches += 1

        keyword_score = keyword_matches / len(keywords) if keywords else 0

        # 类型相关性得分
        type_score = 0
        doc_type = document.get('type', '')
        if any(t in query_lower for t in ['考勤', '上班', '请假']) and doc_type == 'policy':
            type_score = 0.2
        elif any(t in query_lower for t in ['报销', '财务', '预算']) and doc_type == 'finance':
            type_score = 0.2
        elif any(t in query_lower for t in ['工作', '办公', '环境']) and doc_type == 'workplace':
            type_score = 0.2

        # 综合得分
        total_score = exact_match_score + keyword_score * 0.4 + type_score
        return min(1.0, total_score)

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.is_initialized:
                await self.initialize()

            # 测试简单查询
            results = await self.retrieve("测试查询", top_k=1)
            return True
        except Exception as e:
            print(f"本地RAG检索器健康检查失败: {e}")
            return False

    async def add_document(self, content: str, title: str, source: str, doc_type: str = "unknown"):
        """添加新文档到本地知识库"""
        new_doc = {
            'content': content,
            'title': title,
            'source': source,
            'type': doc_type
        }
        self.documents.append(new_doc)
        print(f"已添加新文档: {title}")

    async def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)