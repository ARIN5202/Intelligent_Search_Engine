"""基于 yfinance 的金融行情与历史数据检索器。"""

from __future__ import annotations

from typing import Any, Dict, Optional
import datetime as dt

from config import Settings
from .base_retriever import BaseRetriever, RetrievedDocument


class YFinanceRetriever(BaseRetriever):
    """使用 yfinance 获取实时或历史行情。"""

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
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "YFinanceRetriever requires the yfinance package. Install with: pip install yfinance"
            ) from exc

    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        **_: Any,
    ):
        import yfinance as yf  # type: ignore

        ticker = (symbol or query).strip().upper()
        if not ticker:
            raise ValueError("A ticker symbol is required for finance queries.")

        ticker_obj = yf.Ticker(ticker, session=self._session)

        # Historical path if date range or period is provided
        if start_date or end_date or period:
            hist = ticker_obj.history(
                start=start_date or None,
                end=end_date or None,
                period=period or "max",
                interval=interval,
            )
            if hist.empty:
                raise RuntimeError(f"No historical data returned for {ticker}.")

            # Sort by date descending so we can take the most recent first
            df = hist.sort_index(ascending=False)
            documents = []
            for rank, (idx, row) in enumerate(df.head(top_k).iterrows(), start=1):
                date_str = (
                    idx.strftime("%Y-%m-%d")
                    if isinstance(idx, (dt.datetime, dt.date))
                    else str(idx)
                )
                lines = [f"Symbol: {ticker}", f"Date: {date_str}"]
                for key in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
                    if key in row:
                        lines.append(f"{key}: {row[key]}")

                documents.append(
                    RetrievedDocument(
                        content="\n".join(lines),
                        source="finance_yfinance",
                        score=1.0 / rank,
                        metadata={
                            "mode": "history",
                            "date": date_str,
                            "start_date": start_date,
                            "end_date": end_date,
                            "period": period,
                            "interval": interval,
                            "raw": row.to_dict(),
                        },
                    )
                )

            return documents, {
                "requested_symbol": ticker,
                "mode": "history",
                "dates_returned": [doc.metadata.get("date") for doc in documents],
            }

        # Realtime/last available quote path
        info: Dict[str, Any] = {}
        fast_info = getattr(ticker_obj, "fast_info", None)
        if fast_info:
            info = dict(fast_info)

        price = info.get("last_price")
        prev_close = info.get("previous_close")
        change_value = None
        change_percent = None
        if price is not None and prev_close is not None and prev_close != 0:
            change_value = price - prev_close
            change_percent = (change_value / prev_close) * 100

        # Fallback: try pulling the last close from history if price/prev_close missing
        hist_for_gap = None
        if price is None or prev_close is None:
            try:
                hist_for_gap = ticker_obj.history(period="2d")
            except Exception:
                hist_for_gap = None
        if price is None and hist_for_gap is not None and not hist_for_gap.empty:
            last_row = hist_for_gap.iloc[-1]
            price = float(last_row.get("Close"))
        if prev_close is None and hist_for_gap is not None and len(hist_for_gap) >= 2:
            prev_close = float(hist_for_gap.iloc[-2].get("Close"))

        if change_value is None and price is not None and prev_close not in (None, 0):
            change_value = price - prev_close
            change_percent = (change_value / prev_close) * 100 if prev_close else None

        lines = [f"Symbol: {ticker}"]
        if price is not None:
            lines.append(f"Price: {price:.2f}")
        if change_value is not None:
            lines.append(f"Change: {change_value:+.2f}")
        if change_percent is not None:
            lines.append(f"Change %: {change_percent:+.2f}%")

        doc = RetrievedDocument(
            content="\n".join(lines),
            source="finance_yfinance",
            score=1.0,
            metadata={"raw": info},
        )

        return [doc], {"requested_symbol": ticker, "mode": "quote"}


__all__ = ["YFinanceRetriever"]
