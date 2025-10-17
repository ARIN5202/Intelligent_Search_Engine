"""需要真实网络环境与有效 API Key 的集成测试。

默认情况下，这些测试会被 pytest 跳过；只有当环境变量
``RUN_LIVE_TESTS`` 为 ``"1"`` 时才会执行。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from src.retrieval.manager import RetrievalManager


run_live = os.getenv("RUN_LIVE_TESTS") == "1"
pytestmark = pytest.mark.skipif(not run_live, reason="未设置 RUN_LIVE_TESTS=1，跳过联网集成测试")


@pytest.fixture(scope="module")
def manager() -> RetrievalManager:
    settings = get_settings()
    return RetrievalManager(settings)


def test_finance_alpha_vantage_live(manager: RetrievalManager) -> None:
    result = manager.retrieve_finance("AAPL", provider="finance", top_k=1)
    assert result.documents, "Alpha Vantage 应返回至少一条报价"
    time.sleep(15)  # 避免触发免费层速率限制


def test_finance_yfinance_live(manager: RetrievalManager) -> None:
    result = manager.retrieve_finance("AAPL", provider="finance_yf", top_k=1)
    assert result.documents, "yfinance 应返回至少一条报价"


def test_weather_openweather_live(manager: RetrievalManager) -> None:
    result = manager.retrieve("weather", "Hong Kong", units="metric", top_k=1)
    assert "Weather for" in result.documents[0].content


def test_transport_google_maps_live(manager: RetrievalManager) -> None:
    result = manager.retrieve(
        "transport",
        "Central, Hong Kong",
        destination="The Hong Kong University of Science and Technology",
        mode="transit",
        top_k=1,
    )
    assert result.documents, "Google Maps Directions 应返回至少一条路线"
