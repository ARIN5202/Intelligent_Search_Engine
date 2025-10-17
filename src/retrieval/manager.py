"""Orchestrates all retrieval backends behind a single interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from config import Settings, get_settings
from .retrievers.base_retriever import BaseRetriever, RetrievalResult
from .retrievers.finance_retriever import FinanceRetriever
from .retrievers.local_rag_retriever import LocalRAGRetriever
from .retrievers.transport_retriever import TransportRetriever
from .retrievers.weather_retriever import WeatherRetriever
from .retrievers.web_search_retriever import WebSearchRetriever
from .retrievers.yfinance_retriever import YFinanceRetriever


@dataclass(slots=True)
class RetrievalRequest:
    """Represents a single retrieval call for batch execution."""

    retriever: str
    query: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


class RetrievalManager:
    """High-level facade exposing all retrievers via a unified API."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        auto_register_defaults: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()

        self._retrievers: Dict[str, BaseRetriever] = {}

        if auto_register_defaults:
            self.register_default_retrievers()

    def register_default_retrievers(self) -> None:
        """Instantiate and register the default retrievers."""
        self.register(LocalRAGRetriever(self.settings))
        self.register(WebSearchRetriever(self.settings))
        self.register(WeatherRetriever(self.settings))
        self.register(FinanceRetriever(self.settings))
        self.register(TransportRetriever(self.settings))
        self.register(YFinanceRetriever(self.settings))

    def register(self, retriever: BaseRetriever) -> None:
        """Attach a retriever instance to the manager."""
        self._retrievers[retriever.name] = retriever

    def unregister(self, name: str) -> None:
        """Remove a retriever if it exists (no-op when missing)."""
        self._retrievers.pop(name, None)

    def get_retriever(self, name: str) -> BaseRetriever:
        """Return a retriever instance by name, raising with a helpful error message."""
        try:
            return self._retrievers[name]
        except KeyError as error:
            available = ", ".join(self._retrievers.keys()) or "none"
            raise KeyError(f"Retriever '{name}' is not registered. Available: {available}") from error

    def list_retrievers(self) -> List[str]:
        """Get all registered retriever names for discovery/debugging."""
        return sorted(self._retrievers.keys())

    def has_retriever(self, name: str) -> bool:
        """Check whether a retriever has already been registered."""
        return name in self._retrievers

    def retrieve(self, name: str, query: str, **kwargs: Any) -> RetrievalResult:
        """Invoke the named retriever with the given query and parameters."""
        retriever = self.get_retriever(name)
        return retriever.retrieve(query, **kwargs)

    def retrieve_finance(
        self,
        symbol: str,
        *,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Run a financial lookup, defaulting to yfinance unless a provider is specified."""
        import os

        name = provider or os.getenv("FINANCE_PROVIDER", "finance_yf")
        return self.retrieve(name, symbol, **kwargs)

    def retrieve_batch(self, requests: Sequence[RetrievalRequest]) -> Dict[str, RetrievalResult]:
        """Execute multiple retrievals and return a mapping keyed by retriever name."""
        results: Dict[str, RetrievalResult] = {}
        for request in requests:
            result = self.retrieve(request.retriever, request.query, **request.kwargs)
            results[request.retriever] = result
        return results

    def retrieve_all(
        self,
        query: str,
        *,
        retrievers: Optional[Iterable[str]] = None,
        kwargs_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, RetrievalResult]:
        """Run the same query through multiple retrievers at once."""
        names = retrievers or self._retrievers.keys()
        results: Dict[str, RetrievalResult] = {}
        for name in names:
            extra_kwargs = dict(kwargs_map.get(name, {})) if kwargs_map else {}
            results[name] = self.retrieve(name, query, **extra_kwargs)
        return results


__all__ = ["RetrievalManager", "RetrievalRequest"]
