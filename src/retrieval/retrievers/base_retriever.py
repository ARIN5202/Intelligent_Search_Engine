"""Common abstractions shared by all retrieval backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

from config import Settings


@dataclass(slots=True)
class RetrievedDocument:
    """Represents a single piece of content returned by a retriever."""

    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """Standardised container for retriever responses."""

    query: str
    documents: List[RetrievedDocument]
    provider: str
    latency: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Interface every retriever implementation must follow."""

    def __init__(self, name: str, settings: Settings) -> None:
        self.name = name
        self.settings = settings

    def __call__(self, query: str, **kwargs: Any) -> RetrievalResult:
        return self.retrieve(query, **kwargs)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Validate inputs, measure latency, and delegate to subclass-specific logic."""
        """Execute the retrieval and wrap the result in ``RetrievalResult``."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query must be a non-empty string.")

        top_k = self._validate_top_k(top_k)

        start = time.perf_counter()
        documents, metadata = self._retrieve(
            clean_query,
            top_k=top_k,
            **kwargs,
        )
        elapsed = time.perf_counter() - start

        return RetrievalResult(
            query=clean_query,
            documents=documents[:top_k],
            provider=self.name,
            latency=elapsed,
            metadata=metadata or {},
        )

    @abstractmethod
    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
        **kwargs: Any,
    ) -> Tuple[List[RetrievedDocument], Dict[str, Any]]:
        """Perform backend-specific retrieval and return documents plus metadata."""

    def _validate_top_k(self, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be a positive integer.")
        return min(value, 50)


__all__ = [
    "BaseRetriever",
    "RetrievedDocument",
    "RetrievalResult",
]
