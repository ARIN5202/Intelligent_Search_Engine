"""Weather retriever with Visual Crossing (primary) and OpenWeather current fallback."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Mapping, Optional
import requests
from zoneinfo import ZoneInfo

from config import Settings
from .base_retriever import BaseRetriever, RetrievedDocument


class WeatherRetriever(BaseRetriever):
    """Fetch weather information (current/hourly/daily) for a given location."""

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
        mode: str = "current",  # current | hourly | daily
        at: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        **_: Any,
    ):
        target = (location or query).strip()
        if not target and (lat is None or lon is None):
            raise ValueError("A location or lat/lon must be supplied for weather queries.")

        mode_lower = mode.lower()

        # Prefer Visual Crossing if configured
        if self.settings.visualcrossing_api_key:
            docs, meta = self._retrieve_visualcrossing(
                target=target,
                units=units,
                mode=mode_lower,
                at=at,
            )
            return docs, meta

        # Fallback to OpenWeather current only
        docs, meta = self._retrieve_openweather_current(
            target=target,
            units=units,
            lang=lang,
        )
        return docs, meta

    # ---------- Visual Crossing ----------
    def _retrieve_visualcrossing(
        self,
        *,
        target: str,
        units: str,
        mode: str,
        at: Optional[str],
    ) -> tuple[List[RetrievedDocument], Dict[str, Any]]:
        url = self.settings.visualcrossing_api_url
        api_key = self.settings.visualcrossing_api_key
        params = {
            "unitGroup": "metric" if units == "metric" else "us",
            "include": "hours",
            "key": api_key,
        }
        response = self._session.get(
            f"{url}/{target}",
            params=params,
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()
        data = response.json()

        tz_name = data.get("timezone") or data.get("tz") or data.get("timeZone")
        tzinfo = self._get_tzinfo(tz_name)

        target_dt = self._parse_target_time(at, tzinfo)
        documents: List[RetrievedDocument] = []

        days = data.get("days") or []
        if mode == "daily":
            entry = self._select_day(days, target_dt)
            if entry:
                documents.append(self._build_vc_doc(entry, mode="daily", tzinfo=tzinfo, tz_name=tz_name))
        else:
            # hourly or current -> flatten hours
            hours = self._flatten_hours(days)
            entry = self._select_hour(hours, target_dt)
            if entry:
                documents.append(self._build_vc_doc(entry, mode="hourly", tzinfo=tzinfo, tz_name=tz_name))

        metadata = {
            "provider": "visualcrossing",
            "requested_location": target,
            "mode": mode,
            "at": at,
            "units": units,
            "timezone": tz_name or "UTC",
        }
        return documents, metadata

    @staticmethod
    def _flatten_hours(days: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        hours: List[Mapping[str, Any]] = []
        for d in days:
            hs = d.get("hours") or []
            hours.extend(hs)
        return hours

    @staticmethod
    def _select_day(days: List[Mapping[str, Any]], target_dt: Optional[dt.datetime]) -> Optional[Mapping[str, Any]]:
        if not days:
            return None
        if target_dt is None:
            return days[0]
        tgt = target_dt.date()
        # pick exact date if present, else closest by date diff
        best = min(days, key=lambda d: abs(dt.date.fromisoformat(d["datetime"]) - tgt))
        return best

    @staticmethod
    def _select_hour(hours: List[Mapping[str, Any]], target_dt: Optional[dt.datetime]) -> Optional[Mapping[str, Any]]:
        if not hours:
            return None
        if target_dt is None:
            return hours[0]
        target_ts = target_dt.timestamp()
        best = min(hours, key=lambda h: abs((h.get("datetimeEpoch", target_ts) - target_ts)))
        return best

    @staticmethod
    def _build_vc_doc(entry: Mapping[str, Any], mode: str, tzinfo: Optional[dt.tzinfo], tz_name: Optional[str]) -> RetrievedDocument:
        ts = entry.get("datetimeEpoch")
        if ts:
            base_tz = tzinfo or dt.timezone.utc
            dt_local = dt.datetime.fromtimestamp(ts, base_tz)
            dt_str = dt_local.strftime("%Y-%m-%d %H:%M %Z")
        else:
            dt_str = "N/A"
        conditions = entry.get("conditions")
        temp = entry.get("temp")
        feelslike = entry.get("feelslike")
        humidity = entry.get("humidity")
        wind = entry.get("windspeed")
        precip = entry.get("precip")

        lines = [f"Time: {dt_str}", f"Mode: {mode}"]
        if conditions:
            lines.append(f"Conditions: {conditions}")
        if temp is not None:
            lines.append(f"Temperature: {temp} deg")
        if feelslike is not None:
            lines.append(f"Feels like: {feelslike} deg")
        if humidity is not None:
            lines.append(f"Humidity: {humidity}%")
        if wind is not None:
            lines.append(f"Wind: {wind} m/s")
        if precip is not None:
            lines.append(f"Precip: {precip}")

        return RetrievedDocument(
            content="\n".join(lines),
            source="weather",
            score=1.0,
            metadata={"raw": entry, "dt": ts, "mode": mode, "timezone": tz_name or "UTC"},
        )

    # ---------- OpenWeather fallback (current only) ----------
    def _retrieve_openweather_current(
        self,
        *,
        target: str,
        units: str,
        lang: Optional[str],
    ) -> tuple[List[RetrievedDocument], Dict[str, Any]]:
        api_key = self.settings.weather_api_key
        if not api_key:
            raise RuntimeError("Weather API key missing. Set WEATHER_API_KEY in the environment.")

        params: Dict[str, Any] = {"q": target, "appid": api_key, "units": units}
        if lang:
            params["lang"] = lang
        response = self._session.get(
            self.settings.weather_api_url,
            params=params,
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        doc = self._build_ow_current_doc(payload)
        meta = {"requested_location": target, "units": units, "mode": "current"}
        return [doc], meta

    @staticmethod
    def _build_ow_current_doc(payload: Mapping[str, Any]) -> RetrievedDocument:
        name = payload.get("name", "Unknown location")
        sys = payload.get("sys", {})
        country = sys.get("country")
        weather_entries = payload.get("weather", [])
        descriptions = ", ".join(
            entry.get("description", "").capitalize() for entry in weather_entries if entry
        )
        main = payload.get("main", {})
        wind = payload.get("wind", {})

        lines = [f"Weather for {name}{f', {country}' if country else ''}"]
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

        return RetrievedDocument(
            content="\n".join(lines),
            source="weather",
            score=1.0,
            metadata={"raw": payload},
        )

    # ---------- Utilities ----------
    @staticmethod
    def _parse_target_time(at: Optional[str], tzinfo: Optional[dt.tzinfo]) -> Optional[dt.datetime]:
        if not at:
            return None
        lower = at.lower().strip()
        now = dt.datetime.now(tzinfo or dt.timezone.utc)
        if lower in ("now", "current"):
            return now
        if lower in ("today",):
            return now.replace(hour=12, minute=0, second=0, microsecond=0)
        if "tomorrow" in lower:
            return (now + dt.timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
        if "afternoon" in lower:
            base = now if "tomorrow" not in lower else now + dt.timedelta(days=1)
            return base.replace(hour=15, minute=0, second=0, microsecond=0)
        try:
            return dt.datetime.fromisoformat(at)
        except Exception:
            return None

    @staticmethod
    def _get_tzinfo(tz_name: Optional[str]) -> Optional[dt.tzinfo]:
        if not tz_name:
            return None
        try:
            return ZoneInfo(tz_name)
        except Exception:
            return None


__all__ = ["WeatherRetriever"]
