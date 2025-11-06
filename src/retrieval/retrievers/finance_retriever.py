"""Finance retriever leveraging APIs such as AlphaVantage."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import requests

from config import Settings
from .base_retriever import BaseRetriever, RetrievedDocument


class FinanceRetriever(BaseRetriever):
    """Fetch live market data for equities or indices."""
    domain = [
        "finance", "金融", "财经", "股票", "股价", "股市", "stock", "stock price",
        "投资", "investment", "investing", "财报", "市值", "ticker", "股票代码",
        "analyst", "audit"
    ]

    description = (
        "It is used to obtain real-time stock prices, market capitalis, company information and other financial data of specific stock codes (Ticker symbols, such as 'AAPL' or 'NVDA')."
    )

    def __init__(
        self,
        settings: Settings,
        *,
        session: Optional[requests.Session] = None,
        default_function: str = "GLOBAL_QUOTE",
    ) -> None:
        super().__init__(name="finance", settings=settings)
        self._session = session or requests.Session()
        self._default_function = default_function

    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
        symbol: Optional[str] = None,
        function: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        **_: Any,
    ):
        """Query Alpha Vantage for a quote and map the response to the unified schema."""
        ticker = (symbol or query).strip().upper()
        if not ticker:
            raise ValueError("A ticker symbol is required for finance queries.")

        url = self.settings.finance_api_url
        api_key = self.settings.finance_api_key
        if not api_key:
            raise RuntimeError(
                "Finance API key missing. Set FINANCE_API_KEY in the environment."
            )

        function_name = (function or self._default_function).strip()

        request_params: Dict[str, Any] = {"function": function_name, "symbol": ticker, "apikey": api_key}
        if params:
            request_params.update(params)

        response = self._session.get(
            url,
            params=request_params,
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()

        document = self._build_document(ticker, payload)
        metadata = {
            "requested_symbol": ticker,
            "function": function_name,
        }
        return [document], metadata

    @staticmethod
    def _build_document(symbol: str, payload: Mapping[str, Any]) -> RetrievedDocument:
        """Convert the quote response into a standardised document."""
        quote = FinanceRetriever._extract_quote(payload)
        if not quote:
            raise RuntimeError(f"Finance API response did not contain quote data for {symbol}.")

        price = quote.get("price") or quote.get("05. price")
        change = quote.get("change") or quote.get("09. change")
        percent_change = quote.get("change_percent") or quote.get("10. change percent")

        lines = [f"Symbol: {symbol}"]
        if price:
            lines.append(f"Price: {price}")
        if change:
            lines.append(f"Change: {change}")
        if percent_change:
            lines.append(f"Change %: {percent_change}")

        content = "\n".join(lines)

        return RetrievedDocument(
            content=content or f"Quote data for {symbol}",
            source="finance",
            score=1.0,
            metadata={"raw": payload},
        )

    @staticmethod
    def _extract_quote(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """Find the quote dictionary inside various Alpha Vantage response shapes."""
        for key in ("Global Quote", "Quote", "data"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value  # type: ignore[return-value]
        if isinstance(payload, Mapping):
            return payload  # type: ignore[return-value]
        return None


__all__ = ["FinanceRetriever"]
