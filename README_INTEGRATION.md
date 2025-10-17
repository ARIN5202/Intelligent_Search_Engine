Retrieval 模块集成指南（README_INTEGRATION）
==========================================

本指南面向需要与“检索模块”协作的其他小组成员（主要是 AI 大脑/路由/编排等逻辑负责人）。重点说明模块之间的耦合点、调用方式、输入输出、错误处理，以及在联调前需要达成共识的事项。

1. 总体架构角色
---------------

```
             ┌───────────────────────┐
             │  AI 大脑（LLM Router） │
             └────────┬──────────────┘
                      │
        调用 RetrievalManager 接口（统一入口）
                      │
     ┌────────────────┴────────────────┐
     │                                 │
┌────▼────┐                        ┌───▼─────────────────┐
│ 本地RAG │ 语义检索（LlamaIndex） │ 其余外部数据源      │
└────────┘                        └──────────┬─────────┘
                                              │
   ┌────────────────────────┬─────────────────┼──────────────┬────────────────┐
   │网页搜索（Serper） │天气（OpenWeather）│金融①（Alpha）│金融②（yfinance）│交通（Google）│
   └────────────────────────┴─────────────────┴──────────────┴────────────────┘
```

关键点：
- 外部模块永远与 `RetrievalManager` 交互，不直接触碰各检索器。
- 所有检索返回统一的 `RetrievalResult` 结构，可以直接被路由器 / 重排器 / 回答生成模块消费。
- 金融类检索提供双路线，可通过参数或环境变量选择。

2. 对接契约
----------

### 2.1 `RetrievalManager`（src/retrieval/manager.py:27）

| 方法 | 说明 |
| ---- | ---- |
| `list_retrievers()` | 返回可用检索器名称列表 |
| `retrieve(name, query, **kwargs)` | 调用指定检索器 |
| `retrieve_finance(symbol, provider=None, **kwargs)` | 金融便捷方法，默认使用 `.env` 中的 `FINANCE_PROVIDER`（推荐 `finance_yf`） |
| `retrieve_all(query, retrievers=None, kwargs_map=None)` | 对多个检索器执行同一查询 |
| `retrieve_batch(requests)` | 批量执行不同检索请求 |

调用示例：
```python
from src.retrieval.manager import RetrievalManager

rm = RetrievalManager()
result = rm.retrieve("weather", "Hong Kong", units="metric")

finance_default = rm.retrieve_finance("AAPL")                 # 默认 yfinance
finance_alpha   = rm.retrieve_finance("AAPL", provider="finance")  # Alpha Vantage
```

### 2.2 `RetrievalResult` / `RetrievedDocument` 结构（src/retrieval/retrievers/base_retriever.py）

- `RetrievalResult`
  - `query: str` （清洗后的查询）
  - `documents: List[RetrievedDocument]`
  - `provider: str` （实际使用的检索器名称，如 `weather`, `finance_yf`, `local_rag`）
  - `latency: float` （秒）
  - `metadata: Dict[str, Any]`（检索器的额外信息，如 `{"requested_location": "Hong Kong", ...}`）
- `RetrievedDocument`
  - `content: str` （可读文本）
  - `source: str` （来源标识，RAG 为文件名，外部 API 为供应商）
  - `score: float` （相关度/排序分，语义检索为余弦分，外部 API 可能是置信度或位置）
  - `metadata: Dict[str, Any]` （原始 JSON 信息、URL、doc_id 等）

外部模块只需遍历 `result.documents`，就能拿到标准化的内容和相关度。

3. 常用检索器说明
----------------

| 名称（provider） | 调用方式 | 主要参数 | 典型 metadata | 备注 |
|------------------|----------|----------|---------------|------|
| `local_rag` | `rm.retrieve("local_rag", query, top_k=3)` | `top_k` | `{"persist_dir": ".../storage/llama_index"}` | 语义检索，需提前构建索引 |
| `web_search` | `rm.retrieve("web_search", query, top_k=5)` | `params`（可选）| `{"raw_item_count": N}` | Serper，默认 POST，返回 `url/title/snippet` |
| `weather` | `rm.retrieve("weather", location, units="metric")` | `units`, `lang` | `{"requested_location": "...", "units": "metric"}` | OpenWeather |
| `finance` | `rm.retrieve_finance("AAPL", provider="finance")` | `function`（Alpha Vantage），`symbol` | `{"function": "...", "requested_symbol": "AAPL"}` | 走 Alpha Vantage |
| `finance_yf` | `rm.retrieve_finance("AAPL", provider="finance_yf")` | `symbol` | `{"requested_symbol": "AAPL"}` | yfinance，可能受限流 |
| `transport` | `rm.retrieve("transport", origin, destination=..., mode="transit")` | `destination` 必填，`mode` | `{"origin": "...", "destination": "...", "mode": "transit"}` | Google Directions |

4. 数据流与错误处理
-----------------

### 4.1 成功路径

1. Router 根据用户意图选择调用哪种检索器。
2. 调用 `RetrievalManager.retrieve(...)` 或 `retrieve_finance(...)`。
3. 得到 `RetrievalResult`，可以直接拿 `documents` 进行后续处理（重排/回答生成）。

### 4.2 失败路径 & 建议处理

| 场景 | 现象 | 建议动作 |
| ---- | ---- | -------- |
| API Key 缺失 | `RuntimeError: ... credentials are not configured` | 上层捕获后提示“后端未配置某某服务”或 fallback 到其他数据源 |
| 网络超时 | `requests.exceptions.Timeout` | `BaseRetriever` 会向上抛出异常；上层可重试或切换备选服务 |
| yfinance 限流 | `yfinance.exceptions.YFRateLimitError` | catch 后 fallback 到 Alpha Vantage 或提示用户稍后再试 |
| Google Directions 403 | metadata 中通常无结果，可根据 `metadata` 判断 `mode`/`origin` 等是否需要调整 |
| LlamaIndex 索引缺失 | `FileNotFoundError: persist dir not found` | 需要先执行 `scripts/build_rag_index.py` |

### 4.3 并发与缓存

- `RetrievalManager` 内部没有全局请求缓存；如需缓存策略，请在上层实现。
- `LocalRAGRetriever` 内部使用锁确保索引只加载一次，多线程环境安全。
- 外部 API 若存在速率限制，建议上层实现重试机制或调用节流。

5. 联调前的共识点
---------------

1. **查询格式**
   - `local_rag`: 输入自然语言即可，内部做向量化。
   - `weather`: 建议传城市名称或 “城市, 国家/地区” 格式（例：`"Hong Kong, HK"`）。
   - `transport`: `origin` 与 `destination` 需尽量给出可被 Google 识别的地址字符串。
   - `finance/finance_yf`: 必须传证券代码（示例：`AAPL`, `TSLA`, `0700.HK`）。

2. **返回消费方式**
   - 上层统一读取 `result.documents` 的 `content/score/source`。
   - 在需要原始字段时，访问 `document.metadata`（例如 `metadata["url"]`）。

3. **异常协商**
   - 对于外部 API 的错误（401/403/429/超时），由检索器抛异常；上层决定是否重试/切换/提示。
   - 对于本地索引缺失，需在部署或启动脚本中提前构建。

4. **配置管理**
   - 所有 API key 通过 `.env` 统一管理，不应硬编码。
   - 若要在不同环境部署（开发/测试/生产），请约定 `.env` 变体或使用环境变量覆盖策略。

5. **Schema 扩展**
   - 如需增加新的检索器，请遵守 `BaseRetriever` 协议，实现 `_retrieve` 返回 `RetrievedDocument` 列表，并在 `RetrievalManager.register_default_retrievers()` 中注册。
   - 其他模块若依赖特定 `metadata` 字段，应在文档中记录，以免未来调整解析逻辑时破坏兼容性。

6. 输入输出示例
--------------

### 6.1 本地 RAG

**输入**：
```python
rm.retrieve("local_rag", "公司请假流程", top_k=2)
```

**输出**：
```python
RetrievalResult(
  query="公司请假流程",
  provider="local_rag",
  documents=[
    RetrievedDocument(
      content="... Company leave policy ...",
      source="demo.txt",
      score=0.48,
      metadata={"file_path": "demo.txt", ...}
    ),
    ...
  ],
  metadata={"persist_dir": ".../storage/llama_index"}
)
```

### 6.2 网页搜索（Serper）

**输入**：
```python
rm.retrieve("web_search", "HKUST recent achievements", top_k=3)
```

**输出**：
```python
documents[0].content == "Article Title\nSnippet..."
documents[0].metadata["url"] == "https://..."
```

### 6.3 金融检索

**默认（yfinance）**：
```python
rm.retrieve_finance("AAPL")  # provider 默认为 yfinance
```

**Alpha Vantage**：
```python
rm.retrieve_finance("AAPL", provider="finance")
```

输出内容均类似：
```
Symbol: AAPL
Price: 247.66
Change: 2.39
Change %: 0.97%
```

7. 与其他模块的沟通事项
--------------------

- **路由/决策模块**
  - 需根据用户意图选择合适的 `provider`，例如：
    - 问股票 → `retrieve_finance`（考虑选择 yfinance 或 Alpha Vantage）
    - 问天气 → `retrieve("weather", ...)`
    - 问路线 → `retrieve("transport", origin=..., destination=...)`
    - 查知识库 → `retrieve("local_rag", ...)`
  - 建议在路由器中维护 provider whitelist，避免调用未配置的服务。

- **重排/汇总模块**
  - 可直接读取 `documents` 列表，再结合 LLM 进行重排序与生成。
  - `score` 字段可作为初步排序参考（RAG 为余弦分，其他检索器可能为置信度或自定义值）。
  - `metadata` 里带有原始 JSON，可按需写入生成的中间结构。

- **错误与反馈机制**
  - 强烈建议上层捕获 `RetrievalManager` 抛出的异常，并根据上下文决定 fallback 行为。
  - 对于频繁查询（尤其是 yfinance / Alpha Vantage），需要通过队列/限速防止被封禁。

- **部署注意**
  - 本地 RAG 的索引构建属于离线任务，需要在部署流程中明确执行顺序（参考 `scripts/build_rag_index.py`）。
  - `.env` 中的 key 不应被提交到公共仓库，可通过 secrets 管理或部署脚本注入。

8. FAQ（集成相关）
-----------------

**Q：查询直接调用 `LocalRAGRetriever` 可以吗？**  
A：不建议。请统一通过 `RetrievalManager` 调用，便于未来切换实现、统计监控、加缓存等。

**Q：想同时调用多个检索器，怎么做？**  
A：使用 `retrieve_all(query, retrievers=["local_rag", "web_search"])` 或并行调用 `retrieve_batch`。返回值是 `{provider: RetrievalResult}`。

**Q：如何在系统中切换默认金融数据源？**  
A：编辑 `.env` 中的 `FINANCE_PROVIDER`（`finance_yf` 或 `finance`），或在调用 `retrieve_finance` 时显式传 `provider`。

**Q：如何扩展新的检索器？**  
A：继承 `BaseRetriever`、实现 `_retrieve`，然后在 `RetrievalManager.register_default_retrievers()` 中注册。记得为 dummy/live 测试补充用例。

9. 附录：与 AI 模块协同的 checklist
--------------------------------

- [ ] `.env` 中写好了所有线上 API Key，并在部署环境中生效。
- [ ] `data/` 中的本地文档已更新，重新运行过 `scripts/build_rag_index.py`。
- [ ] dummy 测试（`pytest tests/test_retrievers_dummy.py`）通过。
- [ ] live 测试在必要时运行过（`RUN_LIVE_TESTS=1 pytest tests/test_retrievers_live.py`）。
- [ ] 路由模块了解可用检索器名称（`RetrievalManager().list_retrievers()`）。
- [ ] 上层处理好异常（例如 yfinance 限流、谷歌 API 配额等）。
- [ ] 如需缓存或重试，上层模块已有对应策略。

阅读完本指南后，你应能：
- 在 AI 大脑中正确调用检索模块并解释返回结构；
- 清楚哪些字段可能需要跨模块协商；
- 明白如何处理异常与速率限制；
- 计划未来新增检索器或扩展功能。

如需进一步了解实现细节，请参考 `README.md` 内的完整文件/函数解读。
