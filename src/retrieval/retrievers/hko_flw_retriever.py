"""Retriever for HKO local weather forecast (flw), including tcInfo text."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
import requests

from config import Settings
from .base_retriever import BaseRetriever, RetrievedDocument


class HKOLocalForecastRetriever(BaseRetriever):
    """Fetch local weather forecast (flw) which may contain tcInfo."""

    domain = [
        "tropical cyclone","热带季风"
    ]

    description = (
        "It provides generalSituation and tropical cyclone information if have."
    )

    def __init__(self, settings: Settings, *, session: Optional[requests.Session] = None) -> None:
        super().__init__(name="hko_flw", settings=settings)
        self._session = session or requests.Session()

    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
        **_: Any,
    ):
        params = {"dataType": "flw", "lang": "en"}
        resp = self._session.get(
            self.settings.hko_weather_api_url,
            params=params,
            timeout=self.settings.request_timeout,
        )
        resp.raise_for_status()
        payload = resp.json()

        tc_info = payload.get("tcInfo") or ""
        general = payload.get("generalSituation") or ""
        lines = []
        if general:
            lines.append(f"General Situation: {general}")
        if tc_info:
            lines.append(f"Tropical Cyclone Information: {tc_info}")

        doc = RetrievedDocument(
            content="\n".join(lines) or "No forecast text available.",
            source="hko_flw",
            score=1.0,
            metadata={"raw": payload},
        )
        return [doc], {"dataType": "flw"}


__all__ = ["HKOLocalForecastRetriever"]
