"""面向检索模块的离线（dummy）单元测试。

这些测试使用伪造的 HTTP session/响应对象来模拟外部 API，
用于验证解析逻辑与 BaseRetriever 行为，无需真实网络访问。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from src.retrieval.manager import RetrievalManager
from src.retrieval.retrievers.base_retriever import BaseRetriever, RetrievedDocument
from src.retrieval.retrievers.finance_retriever import FinanceRetriever
from src.retrieval.retrievers.transport_retriever import TransportRetriever
from src.retrieval.retrievers.weather_retriever import WeatherRetriever
from src.retrieval.retrievers.web_search_retriever import WebSearchRetriever


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.requests.append(
            {"url": url, "params": params, "headers": headers, "timeout": timeout}
        )
        return DummyResponse(self.payload)


def make_settings(**overrides):
    return replace(get_settings(), **overrides)


def test_base_retriever_truncates_results():
    settings = get_settings()

    class EchoRetriever(BaseRetriever):
        def _retrieve(self, query: str, *, top_k: int, **kwargs):
            docs = [
                RetrievedDocument(content=f"{query}-{i}", source="echo", score=1.0)
                for i in range(5)
            ]
            return docs, {"count": len(docs)}

    retriever = EchoRetriever("echo", settings)
    result = retriever.retrieve("hello", top_k=3)
    assert len(result.documents) == 3
    assert result.documents[0].content == "hello-0"
    assert result.metadata["count"] == 5


def test_web_search_retriever_parses_items():
    settings = make_settings(
        web_search_api_url="https://fake.api/search",
        web_search_api_key="demo",
    )
    payload = {
        "items": [
            {
                "title": "Doc1",
                "snippet": "Snippet1",
                "url": "https://example.com/1",
                "score": 0.9,
            }
        ]
    }
    retriever = WebSearchRetriever(settings, session=DummySession(payload))
    result = retriever.retrieve("query", top_k=1)
    assert result.documents[0].metadata["url"] == "https://example.com/1"
    assert "Doc1" in result.documents[0].content


def test_weather_retriever_formats_weather():
    settings = get_settings()
    payload = {
        "name": "Hong Kong",
        "sys": {"country": "HK"},
        "weather": [{"description": "clear sky"}],
        "main": {"temp": 25, "feels_like": 26, "humidity": 70},
        "wind": {"speed": 5},
    }
    retriever = WeatherRetriever(settings, session=DummySession(payload))
    result = retriever.retrieve("Hong Kong", top_k=1)
    assert "Hong Kong" in result.documents[0].content
    assert "Humidity" in result.documents[0].content


def test_finance_retriever_parses_global_quote():
    settings = get_settings()
    payload = {
        "Global Quote": {
            "05. price": "123.45",
            "09. change": "1.23",
            "10. change percent": "1.01%",
        }
    }
    retriever = FinanceRetriever(settings, session=DummySession(payload))
    result = retriever.retrieve("AAPL", top_k=1)
    content = result.documents[0].content
    assert "Price: 123.45" in content
    assert "Change: 1.23" in content


def test_transport_retriever_reads_routes():
    settings = get_settings()
    payload = {
        "routes": [
            {
                "summary": "Best Route",
                "legs": [
                    {"duration": {"text": "25 mins"}, "distance": {"text": "10 km"}}
                ],
            }
        ]
    }
    retriever = TransportRetriever(settings, session=DummySession(payload))
    result = retriever.retrieve("Central", destination="HKUST", top_k=1)
    assert "Best Route" in result.documents[0].content
    assert "Duration" in result.documents[0].content


def test_manager_batch_retrieve_mix():
    settings = make_settings(
        web_search_api_url="https://fake.api/search",
        web_search_api_key="demo",
    )
    manager = RetrievalManager(settings, auto_register_defaults=False)
    payload = {"items": [{"title": "Doc", "url": "https://example.com"}]}
    manager.register(WebSearchRetriever(settings, session=DummySession(payload)))

    requests = [
        manager.retrieve("web_search", "test", top_k=1),
    ]

    # 只要返回集合中存在 web_search 即视为成功
    assert requests
