"""需要真实网络环境与有效 API Key 的集成测试。

默认情况下，这些测试会被 pytest 跳过；只有当环境变量
``RUN_LIVE_TESTS`` 为 ``"1"`` 时才会执行。
"""

from __future__ import annotations

import os, logging
import sys
import time
from pathlib import Path

import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from src.retrieval.manager import RetrievalManager


# run_live = os.getenv("RUN_LIVE_TESTS") == "1"
# pytestmark = pytest.mark.skipif(not run_live, reason="未设置 RUN_LIVE_TESTS=1，跳过联网集成测试")

# tests/_pretty.py  (新建一个小工具文件，也可直接写在测试里)
import re, textwrap
from pathlib import Path

def _first_md_heading(text: str) -> str | None:
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None

def _compact(s: str, width: int = 160) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return textwrap.shorten(s, width=width, placeholder="…")

def pretty_print_result(res, top_k: int = 3, preview_chars: int = 160) -> str:
    out = []
    out.append(f"[{res.provider}] {res.query}  (latency={res.latency:.2f}s)")
    for i, d in enumerate(res.documents[:top_k], 1):
        title = _first_md_heading(d.content) or Path(d.source).name
        preview = _compact(d.content, preview_chars)
        out.append(f"{i}. {title}  | score={d.score:.3f}")
        out.append(f"   src: {Path(d.source).name}")  # 只显示文件名，不打印整条绝对路径
        out.append(f"   {preview}")
    return "\n".join(out)

@pytest.fixture(scope="module")
def manager() -> RetrievalManager:
    settings = get_settings()
    return RetrievalManager(settings)


def test_finance_alpha_vantage_live(manager: RetrievalManager) -> None:
    result = manager.retrieve_finance("NVDA", provider="finance", top_k=1)
    print(result)
    print("=" * 40)
    assert result.documents, "Alpha Vantage 应返回至少一条报价"
    time.sleep(15)  # 避免触发免费层速率限制


def test_finance_yfinance_live(manager: RetrievalManager) -> None:
    result = manager.retrieve_finance("NVDA", provider="finance_yf", top_k=1)
    print(result)
    print("=" * 40)
    assert result.documents, "yfinance 应返回至少一条报价"


def test_weather_openweather_live(manager: RetrievalManager) -> None:
    result = manager.retrieve("weather", "Kunming", units="metric", top_k=1)
    print(result)
    print("=" * 40)
    assert "Weather for" in result.documents[0].content


def test_transport_google_maps_live(manager: RetrievalManager) -> None:
    result = manager.retrieve(
        "transport",
        "Central, Hong Kong",
        destination="The Hong Kong University of Science and Technology",
        mode="transit",
        top_k=1,
    )
    print(result)
    print("=" * 40)
    assert result.documents, "Google Maps Directions 应返回至少一条路线"


def test_google_search_live(manager: RetrievalManager) -> None:
    result = manager.retrieve(
        "web_search",
        "中国的首都是哪？",
        top_k=3,
        params={'gl': 'cn', 'hl': 'zh-CN'}
    )
    print(result)
    print("=" * 40)
    assert result.documents, "Google search 至少返回三个结果"

def test_local_rag(manager):
    res = manager.retrieve("local_rag", "What is the Vance Protocol?")
    print(pretty_print_result(res, top_k=3))
    # 断言只做必要检查，避免把整个对象打印出来
    assert res.documents and res.documents[0].score > 0

def test_local_rag2(manager):
    res = manager.retrieve("local_rag", "What is the capital city of the Republic of Sereleia, and on which main island is it located?")
    print(pretty_print_result(res, top_k=3))
    # 断言只做必要检查，避免把整个对象打印出来
    assert res.documents and res.documents[0].score > 0