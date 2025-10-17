## 目录

1. [模块概览：我们在做什么](#模块概览我们在做什么)
2. [先准备环境与依赖](#先准备环境与依赖)
3. [配置：.env 里要写什么](#配置env-里要写什么)
4. [目录结构速读](#目录结构速读)
5. [逐文件、逐函数详细解读](#逐文件逐函数详细解读)
6. [离线构建 + 在线运行：一步步操作](#离线构建--在线运行一步步操作)
7. [测试策略（Dummy & Live）](#测试策略dummy--live)
8. [常见问题 & 调试提示](#常见问题--调试提示)
9. [下一步怎么和队友协同](#下一步怎么和队友协同)

---

## 模块概览：我们在做什么

- **定位**：负责所有数据检索，与 AI 大脑（路由/生成）并行开发。
- **核心角色**：`RetrievalManager` 是唯一的对外入口，内部统一管理多个“检索器”。
- **支持的数据源**：
  1. 本地知识库（Local RAG，基于 LlamaIndex + HuggingFace 嵌入）
  2. 网页搜索（Serper API）
  3. 天气（OpenWeather）
  4. 金融行情（Alpha Vantage & yfinance 双渠道）
  5. 交通路线（Google Maps Directions）
- **统一返回**：所有检索都返回 `RetrievalResult` → `RetrievedDocument` 列表，AI 大脑无需关心底层差异。

---

## 先准备环境与依赖

1. Python 3.10 以上（示例环境为 3.12.3）
2. 建议创建虚拟环境（venv 或 conda）
3. 安装依赖（逐条执行即可）：
   ```bash
   pip install requests pytest
   pip install yfinance
   pip install llama-index-core llama-index-embeddings-huggingface sentence-transformers
   ```
   > 如果遇到 `pip` 提示 `spacy/thinc/numpy` 版本冲突，按提示升级即可：
   > `pip install --upgrade thinc numpy spacy`

---

## 配置：.env 里要写什么

`config.py` 会读取 `.env`，常用变量如下（全部大写）：

| 变量 | 说明 |
| ---- | ---- |
| `BASE_DIR` | 项目根目录（默认当前仓库） |
| `DATA_DIR` | 本地知识库原始文本目录（默认 `./data`） |
| `STORAGE_DIR` | 索引输出目录（默认 `./storage`） |
| `LLAMA_PERSIST_DIR` | LlamaIndex 持久化目录（默认 `storage/llama_index`） |
| `LLAMA_EMBEDDING_MODEL` | Hugging Face 嵌入模型（默认 `intfloat/multilingual-e5-base`） |
| `REQUEST_TIMEOUT` | HTTP 请求超时秒数（默认 10） |
| `WEB_SEARCH_API_URL` | Serper endpoint（`https://google.serper.dev/search`） |
| `WEB_SEARCH_API_KEY` | Serper API Key |
| `WEB_SEARCH_API_METHOD` | `POST`（Serper 要求） |
| `WEB_SEARCH_AUTH_HEADER` | `X-API-KEY` |
| `WEB_SEARCH_AUTH_PREFIX` | 空字符串 |
| `WEATHER_API_URL` / `WEATHER_API_KEY` | OpenWeather |
| `FINANCE_API_URL` / `FINANCE_API_KEY` | Alpha Vantage |
| `FINANCE_PROVIDER` | 默认金融供应商（`finance_yf` 或 `finance`） |
| `TRANSPORT_API_URL` / `TRANSPORT_API_KEY` | Google Maps Directions |
| `USER_AGENT` | 自定义 UA（默认 `IntelligentSearchEngine/1.0`） |
| `RUN_LIVE_TESTS` | （运行 live 测试时）设为 `1` |

示例：
```dotenv
WEB_SEARCH_API_URL=https://google.serper.dev/search
WEB_SEARCH_API_KEY=<serper_key>
WEB_SEARCH_API_METHOD=POST
WEB_SEARCH_AUTH_HEADER=X-API-KEY
WEB_SEARCH_AUTH_PREFIX=

WEATHER_API_URL=https://api.openweathermap.org/data/2.5/weather
WEATHER_API_KEY=<openweather_key>

FINANCE_API_URL=https://www.alphavantage.co/query
FINANCE_API_KEY=<alpha_vantage_key>
FINANCE_PROVIDER=finance_yf

TRANSPORT_API_URL=https://maps.googleapis.com/maps/api/directions/json
TRANSPORT_API_KEY=<google_maps_key>

LLAMA_PERSIST_DIR=storage/llama_index
LLAMA_EMBEDDING_MODEL=intfloat/multilingual-e5-base
```

---

## 目录结构速读

```
.
├── config.py                # 加载 .env，提供 Settings 数据类
├── scripts/
│   └── build_rag_index.py   # 构建 LlamaIndex 索引（离线，一次性）
├── src/
│   └── retrieval/
│       ├── manager.py       # RetrievalManager：统一调用入口
│       └── retrievers/
│           ├── base_retriever.py        # 抽象基类 + 标准返回结构
│           ├── local_rag_retriever.py   # LlamaIndex 语义检索
│           ├── web_search_retriever.py  # Serper 网页搜索
│           ├── weather_retriever.py     # OpenWeather
│           ├── finance_retriever.py     # Alpha Vantage
│           ├── yfinance_retriever.py    # yfinance
│           └── transport_retriever.py   # Google Directions
├── tests/
│   ├── test_retrievers_dummy.py  # 离线伪接口测试
│   └── test_retrievers_live.py   # 在线真实调用测试（默认跳过）
├── README.md                # 本文档
└── README_INTEGRATION.md    # 与其他模块协同指南
```

---

## 逐文件逐函数详细解读

### `config.py`
| 名称 | 类型 | 说明 |
| ---- | ---- | ---- |
| `load_environment` | 函数 | 读取 `.env`，不会覆盖已有环境变量 |
| `_to_path` / `_to_float` | 函数 | 字符串 → `Path` / `float`（容错） |
| `Settings` | 数据类 | 保存路径、API、嵌入模型、超时等 |
| `Settings.from_env` | 类方法 | 使用环境变量构造 `Settings` |
| `Settings.ensure_directories` | 方法 | 创建 `storage` 等所需目录 |
| `get_settings` | 函数 | 读取并缓存 `Settings` |

### `scripts/build_rag_index.py`
1. `parse_args()`：解析命令行参数（数据目录、持久目录、嵌入模型）。
2. `main()`：
   - 清空 `persist_dir`
   - 读取 `data/` 文档
   - 调用 HuggingFace 嵌入（默认 `intfloat/multilingual-e5-base`）
   - 构建 `VectorStoreIndex` 并 `persist()` 到 `storage/llama_index`

### `src/retrieval/manager.py`
- `RetrievalManager.__init__`：加载设置，调用 `register_default_retrievers`
- `register_default_retrievers`：默认注册六个检索器
- `register / unregister / get_retriever / list_retrievers / has_retriever`：检索器管理工具方法
- `retrieve(name, query, **kwargs)`：执行指定检索器，返回 `RetrievalResult`
- `retrieve_finance`：金融便捷方法（默认 yfinance，可传 `provider="finance"` 切到 Alpha Vantage）
- `retrieve_batch`：批量执行不同设置
- `retrieve_all`：同一查询投递多个检索器

### `src/retrieval/retrievers/base_retriever.py`
- `RetrievedDocument`：统一的文档结构（content/source/score/metadata）
- `RetrievalResult`：一次检索结果（含 provider、latency）
- `BaseRetriever.retrieve`：共同逻辑（清洗 query → 校验 top_k → 计时 → 调 `_retrieve`）
- `BaseRetriever._retrieve`：抽象方法，子类实现具体 API/索引调用并返回 `(documents, metadata)`

### `src/retrieval/retrievers/local_rag_retriever.py`
- `__init__`：指定索引持久目录，延迟加载
- `_ensure_index`：从磁盘加载 LlamaIndex，缓存以避免重复加载
- `_retrieve`：调用 `index.as_retriever(similarity_top_k=top_k)`，组装 `RetrievedDocument`

### 其他检索器
| 文件 | 核心函数 | 说明 |
| ----- | -------- | ---- |
| `web_search_retriever.py` | `_retrieve` | Serper API（POST），解析 `organic/items` 列表，生成 `content + url` |
| `weather_retriever.py` | `_retrieve`, `_build_document` | OpenWeather JSON → 温度/湿度/风速描述 |
| `finance_retriever.py` | `_retrieve`, `_build_document`, `_extract_quote` | Alpha Vantage `GLOBAL_QUOTE` → 价格/涨跌额/涨跌幅 |
| `yfinance_retriever.py` | `_retrieve` | yfinance 快速行情，自动处理 `fast_info` 或 `history` |
| `transport_retriever.py` | `_retrieve`, `_build_documents`, `_extract_routes`, `_extract_time_field` | Google Directions → 路线摘要、时长、距离 |

### 测试文件
- `tests/test_retrievers_dummy.py`：伪接口单测（模拟 HTTP），保证解析逻辑正确。
- `tests/test_retrievers_live.py`：真实 API 集成测试（设置 `RUN_LIVE_TESTS=1` 时才运行，含限流等待）。

---

## 离线构建 & 在线运行：一步步操作

1. **准备数据**：把所有课程资料放进 `data/`（支持 `.txt/.md`）。
2. **构建索引**（每次更新 `data/` 都要做一次）：
   ```bash
   python scripts/build_rag_index.py --data-dir data --persist-dir storage/llama_index --embedding-model intfloat/multilingual-e5-base
   ```
3. **快速调用示例**：
   ```python
   from src.retrieval.manager import RetrievalManager

   rm = RetrievalManager()

   rag_result = rm.retrieve("local_rag", "公司请假流程", top_k=3)
   weather_result = rm.retrieve("weather", "Hong Kong", units="metric")
   finance_result = rm.retrieve_finance("AAPL")  # 默认 yfinance
   finance_alpha = rm.retrieve_finance("AAPL", provider="finance")  # Alpha Vantage
   transport_result = rm.retrieve(
       "transport",
       "Central, Hong Kong",
       destination="The Hong Kong University of Science and Technology",
       mode="transit",
   )
   ```

---

## 测试策略（Dummy & Live）

### 1. 离线（Dummy）测试 —— 每次改动都跑
```
pytest tests/test_retrievers_dummy.py
```
- 使用伪 HTTP 会话，验证解析逻辑与标准输出结构。
- 不访问网络，稳定、快捷。

### 2. 在线（Live）集成测试 —— 需要时运行
```
RUN_LIVE_TESTS=1 pytest tests/test_retrievers_live.py
```
- 真实访问 Alpha Vantage、yfinance、OpenWeather、Google Directions。
- 捕获真实返回结构、限流、鉴权问题。
- 默认跳过；只有设置 `RUN_LIVE_TESTS=1` 才执行。

---

## 常见问题 & 调试提示

| 现象 | 可能原因 | 处理建议 |
| ---- | -------- | -------- |
| `FileNotFoundError: ... docstore.json` | 未构建 LlamaIndex 索引 | 重新执行 build 脚本 |
| `RuntimeError: Web search API credentials...` | `.env` 未配置 Serper | 填好 `WEB_SEARCH_*` |
| `YFRateLimitError` | yfinance 限流 | 等几分钟或改用 Alpha Vantage |
| Google Directions `REQUEST_DENIED` | 未启用 API 或未开计费 | 在 GCP 控制台开启 |
| OpenWeather 401/403 | Key 错误或配额问题 | 检查 `.env`、确认 Key |
| pip 依赖冲突 | `spacy/thinc/numpy` 版本不匹配 | 手动 `pip install --upgrade` |

调试小技巧：
- 查看注册的检索器：`print(RetrievalManager().list_retrievers())`
- 快速打印所有设置：`from config import get_settings; print(get_settings())`
- 需要更多日志时，可以在具体检索器内部临时加入 `print` 或 `logging`（完成后记得移除）。

---

## 下一步怎么和队友协同

1. **路由逻辑**：AI 大脑负责决定调用哪个 provider（例：股票 → `retrieve_finance`，天气 → `retrieve("weather", ...)`）。
2. **错误处理**：外层捕获异常（如限流、网络错误），决定重试、提示或换数据源。
3. **API Key 管理**：所有 key 都放 `.env`，不要写入代码仓库。
4. **数据准备**：确保 `data/` 最新，并在部署流程中加入“构建索引”步骤。
5. **文档协作**：更多与外部模块交互的细节，见 `README_INTEGRATION.md`。

