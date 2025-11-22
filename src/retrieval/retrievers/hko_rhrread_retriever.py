"""Retriever for HKO Current Weather Report (rhrread), includes tcmessage."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
import requests

from config import Settings
from .base_retriever import BaseRetriever, RetrievedDocument


class HKORhrreadRetriever(BaseRetriever):
    """Fetch current weather report (rhrread) that contains tcmessage list."""

    domain = [
        "tropical cyclone", "热带季风"
    ]

    description = (
        "It is used to extract tcmessage (tropical cyclone position messages) from current weather report."
    )

    def __init__(self, settings: Settings, *, session: Optional[requests.Session] = None) -> None:
        super().__init__(name="hko_rhrread", settings=settings)
        self._session = session or requests.Session()

    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
        **_: Any,
    ):
        params = {"dataType": "rhrread", "lang": "en"}
        resp = self._session.get(
            self.settings.hko_weather_api_url,
            params=params,
            timeout=self.settings.request_timeout,
        )
        resp.raise_for_status()
        payload = resp.json()

        tc_messages: List[str] = payload.get("tcmessage") or []
        lines: List[str] = []
        if tc_messages:
            lines.append("Tropical Cyclone Messages:")
            for msg in tc_messages[:top_k]:
                lines.append(f"- {msg}")
        else:
            lines.append("No tropical cyclone message available.")

        doc = RetrievedDocument(
            content="\n".join(lines),
            source="hko_rhrread",
            score=1.0,
            metadata={"raw": payload, "tcmessage_count": len(tc_messages)},
        )
        return [doc], {"dataType": "rhrread", "tcmessage_count": len(tc_messages)}


__all__ = ["HKORhrreadRetriever"]
