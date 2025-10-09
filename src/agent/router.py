#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能路由器 - 任务2
分析用户问题，决定调用哪些检索工具
"""

import asyncio
from typing import Dict, List, Any, Optional
from ..prompts.templates import PromptTemplates


class Router:
    """智能路由器，负责分析查询并选择合适的检索工具"""

    def __init__(self):
        """初始化路由器"""
        self.available_tools = {
            'local_rag': {
                'name': '本地知识库',
                'description': '搜索公司内部文档、规章制度等',
                'keywords': ['公司', '规定', '制度', '政策', '内部']
            },
            'web_search': {
                'name': '网络搜索',
                'description': '搜索最新的网络信息',
                'keywords': ['最新', '新闻', '当前', '今天', '搜索']
            },
            'weather': {
                'name': '天气查询',
                'description': '获取天气预报信息',
                'keywords': ['天气', '温度', '下雨', '晴天', '预报']
            },
            'finance': {
                'name': '金融数据',
                'description': '获取股票、汇率等金融信息',
                'keywords': ['股票', '汇率', '金融', '投资', '价格']
            },
            'transport': {
                'name': '交通路线',
                'description': '查询交通路线和时间',
                'keywords': ['路线', '交通', '地铁', '公交', '导航']
            }
        }
        self.prompt_templates = PromptTemplates()

    async def route(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析查询并返回路由结果

        Args:
            query: 用户查询
            context: 可选的上下文信息

        Returns:
            路由结果，包含选中的工具和参数
        """
        # 基于关键词的简单路由逻辑（实际项目中应该使用LLM）
        selected_tools = []
        confidence_scores = {}

        query_lower = query.lower()

        # 检查每个工具的关键词匹配
        for tool_id, tool_info in self.available_tools.items():
            score = 0
            for keyword in tool_info['keywords']:
                if keyword in query_lower:
                    score += 1

            if score > 0:
                selected_tools.append(tool_id)
                confidence_scores[tool_id] = score / len(tool_info['keywords'])

        # 如果没有匹配到任何工具，默认使用本地知识库和网络搜索
        if not selected_tools:
            selected_tools = ['local_rag', 'web_search']
            confidence_scores = {'local_rag': 0.5, 'web_search': 0.5}

        return {
            'query': query,
            'selected_tools': selected_tools,
            'confidence_scores': confidence_scores,
            'top_k': self._determine_top_k(query),
            'reasoning': f"基于关键词匹配选择了工具: {selected_tools}"
        }

    def _determine_top_k(self, query: str) -> int:
        """根据查询复杂度确定检索数量"""
        if len(query.split()) > 10:
            return 15  # 复杂查询需要更多信息
        return 10

    async def health_check(self) -> bool:
        """健康检查"""
        return True

    async def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """获取可用工具列表"""
        return self.available_tools