"""Retriever powered by a persisted LlamaIndex vector index.

本实现从 ``Settings.llama_index_dir`` 加载已持久化的 LlamaIndex 索引，
并通过内置的 retriever 进行相似度检索，返回标准化的 RetrievedDocument。
索引需要先由 ``scripts/build_rag_index.py`` 构建并持久化。
"""

from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from config import Settings
from .base_retriever import BaseRetriever, RetrievedDocument


class LocalRAGRetriever(BaseRetriever):
    """基于 LlamaIndex 的本地语义检索。"""

    def __init__(self, settings: Settings, *, persist_dir: Optional[Path | str] = None) -> None:
        """Initialise retriever with project settings and optional custom index directory."""
        super().__init__(name="local_rag", settings=settings)
        self._persist_dir = Path(persist_dir) if persist_dir else settings.llama_index_dir
        self._index = None
        self._lock = Lock()

    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
        **_: Any,
    ) -> Tuple[List[RetrievedDocument], Dict[str, Any]]:
        """Run a semantic search query against the persisted LlamaIndex store."""
        index = self._ensure_index()

        # Lazy import to avoid importing LlamaIndex unless needed
        from llama_index.core import QueryBundle

        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(QueryBundle(query))

        documents: List[RetrievedDocument] = []
        for i, node_with_score in enumerate(results, start=1):
            node = getattr(node_with_score, "node", None) or getattr(node_with_score, "text_node", None)
            score = float(getattr(node_with_score, "score", 1.0 / i) or 1.0 / i)
            if node is None:
                continue

            content = node.get_content() if hasattr(node, "get_content") else str(getattr(node, "text", ""))
            metadata = dict(getattr(node, "metadata", {}) or {})
            source = metadata.get("source") or metadata.get("file_path") or metadata.get("doc_id") or "llama"

            documents.append(
                RetrievedDocument(
                    content=content,
                    source=str(source),
                    score=score,
                    metadata=metadata,
                )
            )

        meta = {
            "persist_dir": str(self._persist_dir),
        }
        return documents, meta

    def _ensure_index(self):
        """Load and cache the LlamaIndex instance from disk."""
        with self._lock:
            if self._index is not None:
                return self._index

            if not self._persist_dir.exists():
                raise FileNotFoundError(
                    f"LlamaIndex persist dir not found: {self._persist_dir}. "
                    "Please run scripts/build_rag_index.py first."
                )

            from llama_index.core import StorageContext, load_index_from_storage
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            storage_context = StorageContext.from_defaults(persist_dir=str(self._persist_dir))
            embed_model = HuggingFaceEmbedding(model_name=self.settings.llama_embedding_model)
            self._index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
            return self._index


__all__ = ["LocalRAGRetriever"]
