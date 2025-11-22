# Retrieval 模块完整指南

## 1. 模块概览
- 角色：提供多源数据检索（本地 RAG、网页搜索、天气、金融、交通），统一入口 `RetrievalManager`。
- 统一返回：`RetrievalResult`（含 `documents: List[RetrievedDocument]`），上层（AI 大脑）只需消费标准化内容和 metadata。

## 2. 环境与依赖
- Python 3.10+，建议虚拟环境。
- 安装核心依赖：
  ```
  pip install requests pytest
  pip install yfinance
  pip install llama-index-core llama-index-embeddings-huggingface sentence-transformers
  ```
- 如果 pip 报版本冲突（spacy/thinc/numpy），按提示升级即可。

## 3. 配置（.env 关键项）
| 变量 | 说明 |
| ---- | ---- |
| WEB_SEARCH_API_URL / WEB_SEARCH_API_KEY / WEB_SEARCH_API_METHOD / WEB_SEARCH_AUTH_HEADER / WEB_SEARCH_AUTH_PREFIX | Serper 配置（POST + X-API-KEY） |
| WEATHER_API_URL / WEATHER_API_KEY | OpenWeather 当前天气接口（2.5） |
| WEATHER_ONECALL_URL | OpenWeather One Call 3.0（小时/日预报），示例 `https://api.openweathermap.org/data/3.0/onecall` |
| FINANCE_API_URL / FINANCE_API_KEY / FINANCE_PROVIDER | Alpha Vantage & 默认金融供应商（finance_yf 或 finance） |
| TRANSPORT_API_URL / TRANSPORT_API_KEY | Google Maps Directions |
| LLAMA_PERSIST_DIR / LLAMA_EMBEDDING_MODEL | 本地 RAG 持久目录与嵌入模型（例：intfloat/multilingual-e5-base） |
| REQUEST_TIMEOUT | HTTP 超时（秒） |

## 4. 目录速览
```
scripts/
  build_rag_index.py      # 构建 LlamaIndex 索引（离线一次性）
  test_finance_history.py # 金融历史行情示例
src/retrieval/
  manager.py              # RetrievalManager 统一入口
  retrievers/             # 具体检索器
    base_retriever.py     # 抽象基类 + 数据结构
    local_rag_retriever.py
    web_search_retriever.py
    weather_retriever.py
    finance_retriever.py
    yfinance_retriever.py
    transport_retriever.py
tests/
  test_retrievers_dummy.py   # 离线伪接口
  test_retrievers_live.py    # 联网集成（需 RUN_LIVE_TESTS=1）
README_INTEGRATION.md     # 与其他模块协作指引
```

## 5. 核心文件功能
- `config.py`：加载 .env，`Settings` 数据类（路径、API、嵌入模型、超时等）。
- `scripts/build_rag_index.py`：读取 `data/` → Hugging Face 嵌入 → LlamaIndex → 持久化到 `storage/llama_index`。
- `manager.py`：注册/调用各检索器；`retrieve_finance` 可通过 `provider` 切换 yfinance / Alpha。
- `base_retriever.py`：统一契约（`RetrievedDocument`，`RetrievalResult`），`retrieve` 包装 `_retrieve`。
- `local_rag_retriever.py`：加载 LlamaIndex 持久化索引，语义检索。
- `web_search_retriever.py`：Serper（POST），解析 organic/items。
- `weather_retriever.py`：当前天气 + One Call（hourly/daily，可选 `at` 指定时间）。
- `finance_retriever.py`：Alpha Vantage Global Quote + TIME_SERIES 支持（`function`, `target_date`）。
- `yfinance_retriever.py`：实时 + 历史行情（`period/start/end/interval`，返回多条文档）。
- `transport_retriever.py`：Google Directions 路线/时长/距离。

## 6. 构建与调用示例
### 6.1 构建本地 RAG 索引
```
python scripts/build_rag_index.py --data-dir data --persist-dir storage/llama_index --embedding-model intfloat/multilingual-e5-base
```
### 6.2 调用示例
```python
from src.retrieval.manager import RetrievalManager
rm = RetrievalManager()

# 本地 RAG
rag = rm.retrieve("local_rag", "公司请假流程", top_k=3)

# 网页搜索
web = rm.retrieve("web_search", "latest hkust transport policy", top_k=5)

# 天气（当前 / 小时 / 天级）
now = rm.retrieve("weather", "Hong Kong", units="metric")
hourly = rm.retrieve("weather", "Hong Kong", mode="hourly", at="tomorrow 15:00", units="metric")

# 金融（yfinance 实时/历史）
quote_yf = rm.retrieve_finance("NVDA")  # 默认 yfinance
hist_yf = rm.retrieve_finance("NVDA", provider="finance_yf", period="5d", interval="1d", top_k=5)

# 金融（Alpha Vantage，含历史）
quote_alpha = rm.retrieve_finance("AAPL", provider="finance")
hist_alpha = rm.retrieve_finance("AAPL", provider="finance",
                                 function="TIME_SERIES_DAILY_ADJUSTED",
                                 target_date="2024-01-02",
                                 outputsize="full")

# 交通
route = rm.retrieve("transport", "Central, Hong Kong",
                    destination="The Hong Kong University of Science and Technology",
                    mode="transit")
```

## 7. 测试
- 离线：`pytest tests/test_retrievers_dummy.py`
- 在线（需外网和密钥）：`RUN_LIVE_TESTS=1 pytest tests/test_retrievers_live.py`

## 8. 常见问题
- OpenWeather One Call 3.0 需要对应权限/套餐；若 401/403，请在 OpenWeather 控制台开通或换新 Key。
- Alpha Vantage 免费层有速率限制（常见 5 req/min）；yfinance 可能限流，需重试或切换 Alpha。
- 本地 RAG 索引缺失：先运行构建脚本；切换嵌入模型或更新 data 后需重建索引。

## 9. 与其他模块协作
详见 `README_INTEGRATION.md`：输入输出契约、异常处理建议、路由决策点、配置管理约定等。
