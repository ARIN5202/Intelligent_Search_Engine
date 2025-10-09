#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG索引构建脚本
读取data/文件夹中的所有文档，构建向量索引并保存到storage/文件夹
"""

import os
import sys
from pathlib import Path
from typing import List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import Config
from src.agent.parser import DocumentParser


class RAGIndexBuilder:
    """RAG索引构建器"""

    def __init__(self):
        self.config = Config()
        self.parser = DocumentParser()
        self.data_dir = project_root / "data"
        self.storage_dir = project_root / "storage"

        # 确保storage目录存在
        self.storage_dir.mkdir(exist_ok=True)

    def scan_documents(self) -> List[Path]:
        """扫描data目录中的所有文档"""
        documents = []
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                documents.append(file_path)
        return documents

    def build_index(self):
        """构建向量索引"""
        print("开始构建RAG索引...")

        # 扫描文档
        documents = self.scan_documents()
        print(f"发现 {len(documents)} 个文档")

        # 解析文档内容
        all_texts = []
        for doc_path in documents:
            print(f"处理文档: {doc_path.name}")
            try:
                text = self.parser.parse_document(str(doc_path))
                if text.strip():
                    all_texts.append(text)
            except Exception as e:
                print(f"处理文档 {doc_path.name} 时出错: {e}")

        # TODO: 在这里添加向量化和索引构建逻辑
        # 例如使用 FAISS, ChromaDB 或其他向量数据库

        print(f"成功处理 {len(all_texts)} 个文档")
        print("RAG索引构建完成！")

    def update_index(self, file_path: str):
        """更新单个文件的索引"""
        print(f"更新文档索引: {file_path}")
        # TODO: 实现增量更新逻辑


if __name__ == "__main__":
    builder = RAGIndexBuilder()
    builder.build_index()