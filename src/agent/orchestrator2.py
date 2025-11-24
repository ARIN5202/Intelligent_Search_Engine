from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional
import time
import logging

from config import get_settings

from .query_analyzer import QueryAnalyzer        # 负责意图识别、关键词抽取等
from .router import Router                       # 决定调用哪些 retriever
from retrieval.manager import RetrievalManager   # 统一调度具体 retriever

from .reranker import Reranker, RerankResult
from .synthesizer import Synthesizer, SynthesizedResponse

logger = logging.getLogger(__name__)
settings = get_settings()


class AIAgent:
    """
    项目对外唯一的“AI 大脑”入口。

    使用方式（在 main.py 里）：
        agent = AIAgent()
        result = agent.run(user_input)

    其中 user_input 一般来自预处理模块，建议结构：
        {
            "raw_query": 原始用户输入（字符串，可能带错别字）,
            "processed_query": 清洗 / 纠错后的文本（没有就用 raw_query）,
            "attachments": [... 可选，预处理阶段解析好的附件信息 ...],
            "preprocess": {... 预处理阶段的日志信息，可选 ...}
        }
    """

    def __init__(self) -> None:
        # --- 上游模块 ---
        self.query_analyzer = QueryAnalyzer()
        self.router = Router()
        self.retrieval_manager = RetrievalManager()

        self.reranker = Reranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            use_retrieval_score=True,
            alpha=0.7,
            batch_size=16,
        )
        self.synthesizer = Synthesizer(
            deployment_name=settings.azure_deployment_name
            if hasattr(settings, "azure_deployment_name")
            else "gpt-4o",
        )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        完整处理一次用户请求，返回结果字典。

        返回结构建议：
        {
            "answer": 最终回答,
            "sources": [     # 方便前端展示引用信息
                {
                    "source": "local_rag" / "web_search" / "weather" / ...,
                    "score": rerank_score,
                    "metadata": {...},
                    "content": "被引用的原文"
                },
                ...
            ],
            "analysis_results": {... 查询分析结果 ...},
            "routes": [... router 的决策结果 ...],
            "llm_metadata": {... LLM 使用信息、token 数、耗时 ...},
            "timing": {... 各阶段耗时统计 ...},
            "raw_query": ...,
            "processed_query": ...
        }
        """
        t0 = time.time()

        raw_query: str = user_input.get("raw_query") or ""
        processed_query: str = user_input.get("processed_query") or raw_query
        attachments = user_input.get("attachments") or []

        result: Dict[str, Any] = {
            "raw_query": raw_query,
            "processed_query": processed_query,
            "answer": "",
            "sources": [],
            "analysis_results": {},
            "routes": [],
            "llm_metadata": {},
            "timing": {},
            "error": None,
        }

        try:
            # ---------------------------
            # 1. 查询分析
            # ---------------------------
            t1 = time.time()
            analysis = self.query_analyzer.analyze(
                query=processed_query,
                attachments=attachments,
            )
            t2 = time.time()

            # Pydantic 对象转 dict，方便序列化
            if hasattr(analysis, "model_dump"):
                analysis_dict = analysis.model_dump()
            elif hasattr(analysis, "dict"):
                analysis_dict = analysis.dict()
            else:
                # 最差情况：直接 asdict / vars
                try:
                    analysis_dict = asdict(analysis)
                except Exception:
                    analysis_dict = vars(analysis)

            result["analysis_results"] = analysis_dict

            # ---------------------------
            # 2. 工具路由
            # ---------------------------
            routes = self.router.route(
                query=processed_query,
                analysis_results=analysis,
            )
            t3 = time.time()

            # 同样处理成可序列化的列表
            normalized_routes: List[Dict[str, Any]] = []
            for r in routes:
                if hasattr(r, "model_dump"):
                    normalized_routes.append(r.model_dump())
                elif hasattr(r, "dict"):
                    normalized_routes.append(r.dict())
                else:
                    try:
                        normalized_routes.append(asdict(r))
                    except Exception:
                        normalized_routes.append(vars(r))
            result["routes"] = normalized_routes

            # ---------------------------
            # 3. 多源检索
            # ---------------------------
            retrieval_results = []
            for route in routes:
                # 这里根据你们 Route 对象的字段来改名：
                #   route.tool_name: str
                #   route.search_query: Optional[str]
                #   route.params: Dict[str, Any]
                tool_name = getattr(route, "tool_name", None) or getattr(
                    route, "retriever_name", None
                )
                if not tool_name:
                    logger.warning("Route 对象缺少 tool_name/retriever_name 字段，跳过：%s", route)
                    continue

                routed_query = getattr(route, "search_query", None) or processed_query
                params = getattr(route, "params", None) or {}

                try:
                    ret = self.retrieval_manager.retrieve(
                        name=tool_name,
                        query=routed_query,
                        **params,
                    )
                    retrieval_results.append(ret)
                except Exception as e:
                    logger.exception("调用 retriever[%s] 出错：%s", tool_name, e)

            t4 = time.time()

            # ---------------------------
            # 4. 高级重排序
            # ---------------------------
            rerank_result: RerankResult = self.reranker.rerank_from_results(
                query=processed_query,
                retrieval_results=retrieval_results,
                top_k=8,
            )
            t5 = time.time()

            # ---------------------------
            # 5. LLM 合成回答
            # ---------------------------
            synth: SynthesizedResponse = self.synthesizer.synthesize(
                raw_query=raw_query,
                query=processed_query,
                rerank_result=rerank_result,
            )
            t6 = time.time()

            # ---------------------------
            # 6. 组装返回结果
            # ---------------------------
            result["answer"] = synth.answer
            result["sources"] = synth.to_sources()
            result["llm_metadata"] = synth.llm_metadata

            result["timing"] = {
                "total": t6 - t0,
                "analyze": t2 - t1,
                "route": t3 - t2,
                "retrieve": t4 - t3,
                "rerank": t5 - t4,
                "synthesize": t6 - t5,
            }

        except Exception as e:
            # 兜底：整个 pipeline 崩了也要给用户一个合理的错误信息
            logger.exception("AIAgent.run 发生未捕获异常：%s", e)
            result["error"] = str(e)
            if not result["answer"]:
                result["answer"] = "抱歉，系统在处理本次请求时遇到了内部错误，请稍后再试。"

        return result
