"""Retriever that proxies queries to a configurable web search API."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
import requests

from config import Settings
from .base_retriever import BaseRetriever, RetrievedDocument


class WebSearchRetriever(BaseRetriever):
    """Perform real-time web searches via an HTTP API."""

    def __init__(
        self,
        settings: Settings,
        *,
        session: Optional[requests.Session] = None,
        default_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(name="web_search", settings=settings)
        self._session = session or requests.Session()
        self._default_params = dict(default_params or {})

    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        **_: Any,
    ):
        """Call the configured web search API (Serper by default) and format results."""
        url = self.settings.web_search_api_url
        api_key = self.settings.web_search_api_key
        if not url or not api_key:
            raise RuntimeError(
                "Web search API credentials are not configured. "
                "Set WEB_SEARCH_API_URL and WEB_SEARCH_API_KEY in the environment."
            )

        method = self.settings.web_search_api_method
        request_params: Dict[str, Any] = {**self._default_params}
        if params:
            request_params.update(params)

        request_headers = {
            self.settings.web_search_auth_header: f"{self.settings.web_search_auth_prefix}{api_key}"
            if self.settings.web_search_auth_prefix
            else api_key,
            "User-Agent": self.settings.user_agent,
        }
        if headers:
            request_headers.update(headers)

        if method == "POST":
            payload = {"q": query}
            payload.update(request_params)
            request_headers.setdefault("Content-Type", "application/json")
            response = self._session.post(
                url,
                json=payload,
                headers=request_headers,
                timeout=self.settings.request_timeout,
            )
        else:
            request_params.setdefault("q", query)
            request_params.setdefault("count", top_k)
            response = self._session.get(
                url,
                params=request_params,
                headers=request_headers,
                timeout=self.settings.request_timeout,
            )
        response.raise_for_status()
        payload = response.json()

        documents = self._to_documents(payload, top_k)
        metadata = {"raw_item_count": len(documents)}
        return documents, metadata

    def _to_documents(self, payload: Any, limit: int) -> List[RetrievedDocument]:
        items = self._extract_items(payload)
        documents: List[RetrievedDocument] = []

        for position, item in enumerate(items[:limit], start=1):
            url = item.get("url") or item.get("link")
            title = item.get("title") or url or "untitled"
            snippet = item.get("snippet") or item.get("description") or ""
            content = f"{title}\n{snippet}".strip()
            documents.append(
                RetrievedDocument(
                    content=content,
                    source="web",
                    score=float(item.get("score", 1.0 / position)),
                    metadata={
                        "url": url,
                        "position": position,
                        "raw": item,
                    },
                )
            )

        return documents

    @staticmethod
    def _extract_items(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if not isinstance(payload, Mapping):
            return []

        for key in ("items", "data", "results", "value", "organic"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        return []


__all__ = ["WebSearchRetriever"]
