#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI代理总协调器
负责整个问答流程的协调和管理
"""

from typing import Dict, Any, Optional
from .router import Router
from .reranker import Reranker
from .synthesizer import Synthesizer
from ..retrieval.manager import RetrievalManager


class AIAgent:
    """AI代理总协调器"""

    def __init__(self):
        """初始化各个组件"""
        self.router = Router()
        self.retrieval_manager = RetrievalManager()
        self.reranker = Reranker()
        self.synthesizer = Synthesizer()

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理用户查询的完整流程

        Args:
            query: 用户问题
            context: 可选的上下文信息

        Returns:
            包含答案和元数据的字典
        """
        try:
            # 1. 路由 - 决定使用哪些检索器
            print(f"🤖 处理查询: {query}")
            routing_result = await self.router.route(query, context)
            print(f"📍 路由结果: {routing_result['selected_tools']}")

            # 2. 检索 - 获取相关信息
            retrieval_results = await self.retrieval_manager.retrieve_from_multiple(
                query=query,
                retrievers=routing_result['selected_tools'],
                top_k=routing_result.get('top_k', 10)
            )
            print(f"📚 检索到 {len(retrieval_results)} 条信息")

            # 3. 重排序 - 筛选最相关的内容
            reranked_results = await self.reranker.rerank(
                query=query,
                documents=retrieval_results,
                top_k=5
            )
            print(f"🔄 重排序后保留 {len(reranked_results)} 条信息")

            # 4. 生成 - 合成最终答案
            final_answer = await self.synthesizer.generate_answer(
                query=query,
                retrieved_docs=reranked_results,
                context=context
            )
            print("✅ 答案生成完成")

            return {
                'answer': final_answer['answer'],
                'sources': final_answer['sources'],
                'confidence': final_answer.get('confidence', 0.0),
                'routing_info': routing_result,
                'retrieval_count': len(retrieval_results),
                'reranked_count': len(reranked_results)
            }

        except Exception as e:
            print(f"❌ 处理查询时发生错误: {e}")
            return {
                'answer': f"抱歉，处理您的问题时遇到了错误: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'error': str(e)
            }

    async def health_check(self) -> Dict[str, bool]:
        """检查各组件的健康状态"""
        return {
            'router': await self.router.health_check(),
            'retrieval_manager': await self.retrieval_manager.health_check(),
            'reranker': await self.reranker.health_check(),
            'synthesizer': await self.synthesizer.health_check()
        }