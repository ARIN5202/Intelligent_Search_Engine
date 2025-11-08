"""Transportation retriever for route planning and travel time lookups."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence
import requests

from config import Settings
from .base_retriever import BaseRetriever, RetrievedDocument


class TransportRetriever(BaseRetriever):
    """Retrieve transport options between an origin and destination."""
    domain = [
        "transport", "交通", "路线", "导航", "地图", "map", "navigation",
        "怎么去", "怎么走", "how to get to",
        "开车", "驾驶", "driving",
        "公交", "地铁", "public transport",
        "多久", "多远", "how long", "distance",
        "机场", "车站", "airport", "station",
        "出发地", "目的地", "origin", "destination"
    ]

    description = (
        "It is used to query the traffic routes, driving distances and estimated travel times between two locations."
        'It is applicable to questions such as "How to get from A to B?" or "How long does it take to drive?"'
    )

    def __init__(
        self,
        settings: Settings,
        *,
        session: Optional[requests.Session] = None,
        default_mode: str = "driving",
    ) -> None:
        super().__init__(name="transport", settings=settings)
        self._session = session or requests.Session()
        self._default_mode = default_mode

    def _retrieve(
        self,
        query: str,
         *,
        top_k: int,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        mode: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        **_: Any,
    ):
        """Call Google Maps Directions API to obtain candidate routes."""
        if not destination:
            raise ValueError("Destination must be provided for transport queries.")

        origin_text = origin.strip()
        if not origin_text:
            raise ValueError("Origin must be provided or inferred from the query.")

        url = self.settings.transport_api_url
        api_key = self.settings.transport_api_key
        if not url or not api_key:
            raise RuntimeError(
                "Transport API credentials missing. Set TRANSPORT_API_URL and TRANSPORT_API_KEY."
            )

        request_params: Dict[str, Any] = {
            "origin": origin_text,
            "destination": destination,
            "mode": mode or self._default_mode,
            "key": api_key,
        }
        if params:
            request_params.update(params)

        response = self._session.get(
            url,
            params=request_params,
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()

        documents = self._build_documents(payload, top_k)
        metadata = {
            "origin": origin_text,
            "destination": destination,
            "mode": request_params["mode"],
        }
        return documents, metadata

    def _build_documents(self, payload: Mapping[str, Any], limit: int) -> List[RetrievedDocument]:
        """Transform Google Directions JSON into `RetrievedDocument` objects."""
        routes = self._extract_routes(payload)
        documents: List[RetrievedDocument] = []
        for idx, route in enumerate(routes[:limit], start=1):
            summary = route.get("summary") or route.get("name") or f"Route {idx}"
            legs: Sequence[Mapping[str, Any]] = route.get("legs") or []
            first_leg = legs[0] if legs else {}

            duration = self._extract_time_field(first_leg.get("duration"))
            distance = self._extract_time_field(first_leg.get("distance"))

            lines = [f"{summary}"]
            if duration:
                lines.append(f"Duration: {duration}")
            if distance:
                lines.append(f"Distance: {distance}")

            content = "\n".join(lines)

            documents.append(
                RetrievedDocument(
                    content=content or summary,
                    source="transport",
                    score=1.0 / idx,
                    metadata={"raw": route, "index": idx},
                )
            )

        return documents

    @staticmethod
    def _extract_routes(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        """Return a list of route dictionaries from the API payload."""
        if "routes" in payload and isinstance(payload["routes"], list):
            return [route for route in payload["routes"] if isinstance(route, Mapping)]
        if isinstance(payload, Mapping):
            return [payload]
        return []

    @staticmethod
    def _extract_time_field(value: Any) -> Optional[str]:
        """Normalise duration/distance fields regardless of their representation."""
        if isinstance(value, Mapping):
            text = value.get("text")
            if isinstance(text, str):
                return text
            value = value.get("value")
        if isinstance(value, (int, float)):
            return f"{value}"
        if isinstance(value, str):
            return value
        return None


__all__ = ["TransportRetriever"]
