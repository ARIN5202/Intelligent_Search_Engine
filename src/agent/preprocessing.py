#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理模块 - 任务1
负责处理各种格式的输入数据的预处理工作
"""

from typing import List, Dict, Any, Optional
import re


class TextPreprocessor:
    """文本预处理器"""

    def __init__(self):
        """初始化预处理器"""
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self) -> set:
        """加载停用词表"""
        # 简单的中文停用词表
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '为',
            '与', '及', '等', '从', '被', '把', '将', '对', '向', '以'
        }
        return stop_words

    def clean_text(self, text: str) -> str:
        """
        清理文本

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)

        # 移除特殊字符（保留中英文、数字和基本标点）
        text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？、；：""''（）《》\[\].,!?;:\'"()\-]', '', text)

        # 统一标点符号
        text = text.replace('，', ', ')
        text = text.replace('。', '. ')
        text = text.replace('！', '! ')
        text = text.replace('？', '? ')

        return text.strip()

    def remove_stop_words(self, text: str) -> str:
        """
        移除停用词

        Args:
            text: 输入文本

        Returns:
            移除停用词后的文本
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def normalize_whitespace(self, text: str) -> str:
        """标准化空白字符"""
        return re.sub(r'\s+', ' ', text).strip()

    def extract_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 使用正则表达式分割句子
        sentences = re.split(r'[。！？.!?]+', text)

        # 清理并过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def truncate_text(self, text: str, max_length: int = 500, preserve_sentences: bool = True) -> str:
        """
        截断文本到指定长度

        Args:
            text: 输入文本
            max_length: 最大长度
            preserve_sentences: 是否保持句子完整性

        Returns:
            截断后的文本
        """
        if len(text) <= max_length:
            return text

        if preserve_sentences:
            sentences = self.extract_sentences(text)
            result = []
            current_length = 0

            for sentence in sentences:
                if current_length + len(sentence) <= max_length:
                    result.append(sentence)
                    current_length += len(sentence)
                else:
                    break

            return '。'.join(result) + '。' if result else text[:max_length]
        else:
            return text[:max_length]

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        将文本分块（用于向量化）

        Args:
            text: 输入文本
            chunk_size: 每个块的大小
            overlap: 块之间的重叠字符数

        Returns:
            文本块列表
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # 如果不是最后一块，尝试在句子边界处断开
            if end < len(text):
                # 在附近寻找句子结束符
                for punctuation in ['。', '！', '？', '.', '!', '?']:
                    last_punct = text.rfind(punctuation, start, end)
                    if last_punct != -1 and last_punct > start + chunk_size // 2:
                        end = last_punct + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap if end < len(text) else end

        return chunks

    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        预处理用户查询

        Args:
            query: 用户查询

        Returns:
            预处理结果
        """
        # 清理查询
        cleaned_query = self.clean_text(query)

        # 提取关键词（移除停用词）
        keywords = self.remove_stop_words(cleaned_query)

        # 分析查询类型
        query_type = self._analyze_query_type(query)

        return {
            'original': query,
            'cleaned': cleaned_query,
            'keywords': keywords,
            'type': query_type,
            'length': len(query)
        }

    def _analyze_query_type(self, query: str) -> str:
        """分析查询类型"""
        query_lower = query.lower()

        # 定义查询类型的关键词
        if any(word in query_lower for word in ['是什么', '什么是', '定义', '解释']):
            return 'definition'
        elif any(word in query_lower for word in ['如何', '怎么', '怎样', '方法']):
            return 'how_to'
        elif any(word in query_lower for word in ['为什么', '原因', '理由']):
            return 'why'
        elif any(word in query_lower for word in ['比较', '对比', '区别', '差异']):
            return 'comparison'
        elif any(word in query_lower for word in ['列出', '列举', '有哪些']):
            return 'list'
        else:
            return 'general'


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self):
        """初始化数据预处理器"""
        self.text_preprocessor = TextPreprocessor()

    def preprocess_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        预处理文档

        Args:
            content: 文档内容
            metadata: 文档元数据

        Returns:
            预处理后的文档数据
        """
        # 清理文本
        cleaned_content = self.text_preprocessor.clean_text(content)

        # 分块处理
        chunks = self.text_preprocessor.chunk_text(cleaned_content)

        # 提取摘要（取前两个句子）
        sentences = self.text_preprocessor.extract_sentences(cleaned_content)
        summary = '。'.join(sentences[:2]) + '。' if len(sentences) >= 2 else cleaned_content[:200]

        return {
            'content': cleaned_content,
            'chunks': chunks,
            'summary': summary,
            'metadata': metadata or {},
            'length': len(cleaned_content),
            'num_chunks': len(chunks)
        }

    def preprocess_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量预处理文档

        Args:
            documents: 文档列表

        Returns:
            预处理后的文档列表
        """
        processed_docs = []

        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})

            processed_doc = self.preprocess_document(content, metadata)
            processed_docs.append(processed_doc)

        return processed_docs