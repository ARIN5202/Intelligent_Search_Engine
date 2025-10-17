"""Weather retriever backed by a configurable HTTP API (OpenWeather by default)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import requests

from config import Settings
from .base_retriever import BaseRetriever, RetrievedDocument


class WeatherRetriever(BaseRetriever):
    """Fetch current weather information for a given location."""

    def __init__(
        self,
        settings: Settings,
        *,
        session: Optional[requests.Session] = None,
    ) -> None:
        super().__init__(name="weather", settings=settings)
        self._session = session or requests.Session()

    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
        location: Optional[str] = None,
        units: str = "metric",
        lang: Optional[str] = None,
        **_: Any,
    ):
        """Fetch current weather information from OpenWeather and summarise key metrics."""
        target = (location or query).strip()
        if not target:
            raise ValueError("A location must be supplied for weather queries.")

        url = self.settings.weather_api_url
        api_key = self.settings.weather_api_key
        if not api_key:
            raise RuntimeError(
                "Weather API key missing. Set WEATHER_API_KEY in the environment."
            )

        params: Dict[str, Any] = {
            "q": target,
            "appid": api_key,
            "units": units,
        }
        if lang:
            params["lang"] = lang

        response = self._session.get(
            url,
            params=params,
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()
        data = response.json()

        document = self._build_document(data)
        metadata = {"requested_location": target, "units": units}
        return [document], metadata

    @staticmethod
    def _build_document(payload: Mapping[str, Any]) -> RetrievedDocument:
        """Format raw weather payload into a human-readable document."""
        location_name = payload.get("name", "Unknown location")
        sys = payload.get("sys", {})
        country = sys.get("country")
        weather_entries = payload.get("weather", [])
        descriptions = ", ".join(
            entry.get("description", "").capitalize() for entry in weather_entries if entry
        )

        main = payload.get("main", {})
        wind = payload.get("wind", {})

        lines = [f"Weather for {location_name}{f', {country}' if country else ''}"]
        if descriptions:
            lines.append(f"Conditions: {descriptions}")
        if "temp" in main:
            lines.append(f"Temperature: {main['temp']} deg")
        if "feels_like" in main:
            lines.append(f"Feels like: {main['feels_like']} deg")
        if "humidity" in main:
            lines.append(f"Humidity: {main['humidity']}%")
        if "speed" in wind:
            lines.append(f"Wind: {wind['speed']} m/s")

        content = "\n".join(lines)

        return RetrievedDocument(
            content=content,
            source="weather",
            score=1.0,
            metadata={"raw": payload},
        )


__all__ = ["WeatherRetriever"]
