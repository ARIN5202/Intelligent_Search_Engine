"""Retriever for HKO weather warning summary (warnsum), including TC8 signals."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
import requests

from config import Settings
from .base_retriever import BaseRetriever, RetrievedDocument


class HKOWarnSumRetriever(BaseRetriever):
    """Fetch HKO warning summary (warnsum), surface tropical cyclone warning signals."""

    domain = [
        "tropical", "warning", "8", "typhone", "台风", "警告", "八号", "风球", "fire"
    ]

    description = (
        "It is a Quick view of current warning summary, especially WTCSGNL (tropical cyclone warning signal, e.g., TC8*)."
    )

    def __init__(self, settings: Settings, *, session: Optional[requests.Session] = None) -> None:
        super().__init__(name="hko_warnsum", settings=settings)
        self._session = session or requests.Session()

    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
        **_: Any,
    ):
        params = {"dataType": "warnsum", "lang": "en"}
        resp = self._session.get(
            self.settings.hko_weather_api_url,
            params=params,
            timeout=self.settings.request_timeout,
        )
        resp.raise_for_status()
        payload = resp.json()

        # payload is expected to be a mapping with keys like WTCSGNL, WTSIG, etc.
        tc_signal = payload.get("WTCSGNL")
        lines: List[str] = []
        if tc_signal:
            lines.append(f"TC Warning Signal: {tc_signal}")

        for key, val in payload.items():
            if key == "WTCSGNL":
                continue
            lines.append(f"{key}: {val}")

        doc = RetrievedDocument(
            content="\n".join(lines) or "No warning summary available.",
            source="hko_warnsum",
            score=1.0,
            metadata={"raw": payload},
        )
        return [doc], {"dataType": "warnsum", "tc_signal": tc_signal}


__all__ = ["HKOWarnSumRetriever"]
