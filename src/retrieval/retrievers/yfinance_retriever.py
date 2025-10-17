"""基于 yfinance 的金融行情检索器。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config import Settings
from .base_retriever import BaseRetriever, RetrievedDocument


class YFinanceRetriever(BaseRetriever):
    """使用 yfinance 快速获取股票或基金的最新行情。"""

    def __init__(
        self,
        settings: Settings,
        *,
        session: Optional[Any] = None,
    ) -> None:
        super().__init__(name="finance_yf", settings=settings)
        self._session = session
        self._ensure_dependency()

    def _ensure_dependency(self) -> None:
        """Ensure yfinance is installed; provide a clear message if missing."""
        try:
            import yfinance  # type: ignore # noqa: F401
        except ImportError as exc:  # pragma: no cover - 运行时错误提示
            raise RuntimeError(
                "使用 YFinanceRetriever 需要先安装 yfinance 库：pip install yfinance"
            ) from exc

    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
        symbol: Optional[str] = None,
        **_: Any,
    ):
        """Fetch the latest quote using yfinance (without network credentials)."""
        import yfinance as yf  # type: ignore

        ticker = (symbol or query).strip().upper()
        if not ticker:
            raise ValueError("必须提供有效的证券代码。")

        ticker_obj = yf.Ticker(ticker, session=self._session)
        fast_info = getattr(ticker_obj, "fast_info", None)
        info: Dict[str, Any] = {}

        if fast_info:
            info = dict(fast_info)

        price = info.get("last_price")
        change = info.get("last_price") and info.get("previous_close")
        change_value = None
        change_percent = None

        if price is not None and info.get("previous_close") is not None:
            change_value = price - info["previous_close"]
            change_percent = (
                (change_value / info["previous_close"]) * 100 if info["previous_close"] else None
            )

        if price is None:
            live_price = ticker_obj.history(period="1d")["Close"]
            if not live_price.empty:
                price = float(live_price.iloc[-1])

        lines = [f"Symbol: {ticker}"]
        if price is not None:
            lines.append(f"Price: {price:.2f}")
        if change_value is not None:
            lines.append(f"Change: {change_value:+.2f}")
        if change_percent is not None:
            lines.append(f"Change %: {change_percent:+.2f}%")

        document = RetrievedDocument(
            content="\n".join(lines),
            source="finance_yfinance",
            score=1.0,
            metadata={"raw": info},
        )

        return [document], {"requested_symbol": ticker}


__all__ = ["YFinanceRetriever"]
