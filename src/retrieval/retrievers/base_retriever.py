#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础检索器接口
定义所有检索器必须实现的接口规范
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseRetriever(ABC):
    """检索器基类，定义统一接口"""

    def __init__(self, name: str = "BaseRetriever"):
        """
        初始化检索器

        Args:
            name: 检索器名称
        """
        self.name = name
        self.description = "基础检索器"
        self.is_initialized = False

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        检索相关信息

        Args:
            query: 查询内容
            top_k: 返回结果数量
            **kwargs: 其他参数

        Returns:
            检索结果列表，每个结果包含:
            - content: 内容文本
            - title: 标题（可选）
            - source: 来源（可选）
            - score: 相关性分数（可选）
            - metadata: 其他元数据（可选）
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            检索器是否正常工作
        """
        pass

    async def initialize(self) -> bool:
        """
        初始化检索器

        Returns:
            是否初始化成功
        """
        try:
            await self._do_initialize()
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"检索器 {self.name} 初始化失败: {e}")
            self.is_initialized = False
            return False

    async def _do_initialize(self):
        """子类可重写的初始化逻辑"""
        pass

    def _format_result(self, content: str, title: str = "", source: str = "", score: float = 0.0,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        格式化检索结果

        Args:
            content: 内容文本
            title: 标题
            source: 来源
            score: 相关性分数
            metadata: 其他元数据

        Returns:
            格式化后的结果字典
        """
        result = {
            'content': content.strip(),
            'title': title.strip(),
            'source': source.strip(),
            'score': max(0.0, min(1.0, score)),  # 确保分数在0-1之间
            'metadata': metadata or {}
        }

        # 添加时间戳
        import time
        result['retrieved_at'] = time.time()
        result['retriever_name'] = self.name

        return result

    def _extract_keywords(self, query: str) -> List[str]:
        """
        从查询中提取关键词

        Args:
            query: 查询内容

        Returns:
            关键词列表
        """
        # 简单的关键词提取（实际项目中可以使用更复杂的NLP方法）
        import re

        # 移除标点符号并分词
        words = re.findall(r'\b\w+\b', query.lower())

        # 过滤停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很',
                      '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]

        return keywords

    async def update_config(self, config: Dict[str, Any]):
        """
        更新配置（子类可重写）

        Args:
            config: 新配置
        """
        print(f"检索器 {self.name} 配置已更新: {config}")