#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索管理器 - 任务3
统一管理所有检索器的调度中心
"""

import asyncio
from typing import Dict, List, Any, Optional
from .retrievers.base_retriever import BaseRetriever
from .retrievers.local_rag_retriever import LocalRAGRetriever
from .retrievers.web_search_retriever import WebSearchRetriever
from .retrievers.weather_retriever import WeatherRetriever
from .retrievers.finance_retriever import FinanceRetriever
from .retrievers.transport_retriever import TransportRetriever


class RetrievalManager:
    """检索管理器，统一管理所有检索器"""

    def __init__(self):
        """初始化检索管理器"""
        self.retrievers: Dict[str, BaseRetriever] = {}
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        """初始化所有检索器"""
        self.retrievers = {
            'local_rag': LocalRAGRetriever(),
            'web_search': WebSearchRetriever(),
            'weather': WeatherRetriever(),
            'finance': FinanceRetriever(),
            'transport': TransportRetriever()
        }

    async def retrieve_from_single(self, query: str, retriever_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        从单个检索器获取信息

        Args:
            query: 查询内容
            retriever_name: 检索器名称
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"检索器不存在: {retriever_name}")

        retriever = self.retrievers[retriever_name]
        try:
            results = await retriever.retrieve(query, top_k)
            # 为结果添加来源标识
            for result in results:
                result['retriever'] = retriever_name
            return results
        except Exception as e:
            print(f"检索器 {retriever_name} 出错: {e}")
            return []

    async def retrieve_from_multiple(self, query: str, retrievers: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        从多个检索器并行获取信息

        Args:
            query: 查询内容
            retrievers: 检索器名称列表
            top_k: 每个检索器返回的结果数量

        Returns:
            合并后的检索结果列表
        """
        # 创建并行任务
        tasks = []
        for retriever_name in retrievers:
            if retriever_name in self.retrievers:
                task = self.retrieve_from_single(query, retriever_name, top_k)
                tasks.append(task)
            else:
                print(f"警告: 检索器不存在 {retriever_name}")

        if not tasks:
            return []

        # 并行执行所有检索任务
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果
        all_results = []
        for i, results in enumerate(results_list):
            if isinstance(results, Exception):
                print(f"检索任务 {i} 失败: {results}")
                continue
            all_results.extend(results)

        return all_results

    async def retrieve_with_fallback(self, query: str, primary_retrievers: List[str], fallback_retrievers: List[str],
                                     top_k: int = 10) -> List[Dict[str, Any]]:
        """
        带降级策略的检索

        Args:
            query: 查询内容
            primary_retrievers: 主要检索器列表
            fallback_retrievers: 备用检索器列表
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        # 首先尝试主要检索器
        results = await self.retrieve_from_multiple(query, primary_retrievers, top_k)

        # 如果主要检索器结果不足，使用备用检索器
        if len(results) < top_k // 2:
            print("主要检索器结果不足，启用备用检索器")
            fallback_results = await self.retrieve_from_multiple(query, fallback_retrievers, top_k)
            results.extend(fallback_results)

        return results[:top_k]

    async def health_check(self) -> Dict[str, bool]:
        """检查所有检索器的健康状态"""
        health_status = {}

        tasks = []
        for name, retriever in self.retrievers.items():
            task = retriever.health_check()
            tasks.append((name, task))

        for name, task in tasks:
            try:
                status = await task
                health_status[name] = status
            except Exception as e:
                print(f"检索器 {name} 健康检查失败: {e}")
                health_status[name] = False

        return health_status

    def get_retriever_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有检索器的信息"""
        info = {}
        for name, retriever in self.retrievers.items():
            info[name] = {
                'name': name,
                'description': getattr(retriever, 'description', ''),
                'is_available': True  # 可以根据实际情况检查
            }
        return info

    async def update_retriever_config(self, retriever_name: str, config: Dict[str, Any]):
        """更新检索器配置"""
        if retriever_name not in self.retrievers:
            raise ValueError(f"检索器不存在: {retriever_name}")

        retriever = self.retrievers[retriever_name]
        if hasattr(retriever, 'update_config'):
            await retriever.update_config(config)
        else:
            print(f"检索器 {retriever_name} 不支持配置更新")