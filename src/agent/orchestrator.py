#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent Orchestrator
Coordinates and manages the entire QA workflow as the brain of the system/APP
Using LangChain to orchestrate the components
"""

import time
from typing import Dict, Any
from .query_analyzer import QueryAnalyzer
from .router import Router
from .reranker import Reranker
from .synthesizer import Synthesizer
from ..retrieval.manager import retrieval_manager
from config import get_settings
from .reranker import RerankResult

settings = get_settings()


class AIAgent:
    """AI Agent Orchestrator"""

    def __init__(self):
        """Initialize pipeline components."""
        self.query_analyzer = QueryAnalyzer()
        self.router = Router()
        self.retrieval_manager = retrieval_manager
        self.reranker = Reranker()
        self.synthesizer = Synthesizer(deployment_name="gpt-4o")

    def _retrieve_documents(
            self,
            routing_results: Dict[str, Any],
            analysis_results: Dict[str, Any],
    ) -> RerankResult:
        """Retrieve documents based on routing results and rerank them."""
        start = time.time()

        # 安全一点地拿 query，避免 key 不存在直接 KeyError
        query = (
                analysis_results.get("rewritten_query")
                or analysis_results.get("query")
                or ""
        )

        try:
            selected_tools = routing_results["selected_tools"]
            retrieval_metadata = routing_results.get("retrieval_metadata", {})  # 防御性一点
            retrieval_results = None
            skip_web_search = False
            k = 10  # Default top_k for web search
            final_usage = None
            tool_retrievals = []

            # Call specialized retrievers if any selected
            if len(selected_tools) > 1:
                try:
                    specialized_tool = selected_tools[0]
                    if specialized_tool == "transport":
                        tool_retrievals.append(self.retrieval_manager.retrieve(
                            name="transport",
                            query=query,
                            origin=retrieval_metadata.get("origin", None),
                            destination=retrieval_metadata.get("destination", None),
                            mode=retrieval_metadata.get("transit_mode", 'transit'),
                            top_k=10,
                        ))      
                    elif specialized_tool == "finance":
                        for ticker in retrieval_metadata["ticker_symbols"]:
                            # if in shenzhen, add these
                            # import os
                            # proxy = 'http://127.0.0.1:7890'
                            # os.environ['HTTP_PROXY'] = proxy
                            # os.environ['HTTPS_PROXY'] = proxy
                            tool_retrievals.append(self.retrieval_manager.retrieve(
                                name="finance_yf",
                                query=query,
                                symbol=ticker,
                                period=retrieval_metadata.get("period", None),
                                start_date=retrieval_metadata.get("start_date", None),
                                end_date=retrieval_metadata.get("end_date", None),
                                top_k=10,
                            ))
                        if tool_retrievals:
                            k = 3    
                    elif specialized_tool == "weather":
                        tool_retrievals.append(self.retrieval_manager.retrieve(
                            name="weather",
                            query=query,
                            location=retrieval_metadata.get("location", None),
                            mode=retrieval_metadata.get("mode", "daily"),   
                            at=retrieval_metadata.get("target_time", None),
                            top_k=10,
                        ))
                        if tool_retrievals:
                            k = 2
                    elif specialized_tool == "hko_warnsum":
                        tool_retrievals.append(self.retrieval_manager.retrieve(
                            name="hko_warnsum",
                            query=query,
                            top_k=10,
                        ))
                        if tool_retrievals:
                            skip_web_search = True
                    elif specialized_tool == "local_rag":
                        tool_retrievals.append(self.retrieval_manager.retrieve(
                            name="local_rag",
                            query=query,
                            top_k=10,
                        ))
                        if tool_retrievals:
                            k = 5

                except Exception as e:
                    print(f"  - No available relevant documents using specialized '{specialized_tool}' retriever")
                    skip_web_search = False
            
            # Always call web search as fallback unless skipped
            if not skip_web_search:
                # Call web search retriever as the base
                retrieval_results = self.retrieval_manager.retrieve(
                    name="web_search",
                    query=query,
                    top_k=k,
                )
                for retrieval in tool_retrievals:
                    retrieval_results.documents.extend(retrieval.documents)
                if not tool_retrievals:
                    final_usage = "web_search"
                else:
                    final_usage = ', '.join(selected_tools)
                print(
                f"  - Retrieved {len(retrieval_results.documents)} documents "
                f"using '{final_usage}' retriever(s)"
                )
            else:
                if retrieval_results is None:
                    retrieval_results = tool_retrievals[0]
                    for retrieval in tool_retrievals[1:]:
                        retrieval_results.documents.extend(retrieval.documents)
                final_usage = specialized_tool
                print(
                f"  - Retrieved {len(retrieval_results.documents)} documents "
                f"using '{final_usage}' retriever(s)"
                )

            # Step 4: rerank the retrieved documents
            rerank_results: RerankResult = self.reranker.rerank_from_results(
                query=query,
                retrieval_results=[retrieval_results],
                top_k=5,
            )

            return rerank_results

        except Exception as e:
            tool_name = routing_results.get("selected_tools", ["unknown"])
            print(f"⚠️ Retrieval and reranking failed for '{tool_name}': {e}")
            # 如果你想看更详细 trace，可以打开下面这行
            # print(traceback.format_exc())

            # ❗关键：异常时仍然返回 RerankResult，而不是 dict
            return RerankResult(
                query=query,
                contexts=[],  # 空上下文
            )

    def run(self, user_input: Dict[str, Any]) -> Any:
        """
        Orchestrate the  pipeline

        Args:
            user_input: Dictionay containing:
                - raw_query: Original user query string
                - attachments: list of structured attachment data
                - attachment_issues: list of issues identified during preprocessing

        Returns:
            Final answer generated by the AI agent
        """
        try:
            print("\n" + "=" * 80)
            print("🚀 Starting Query Processing Pipeline")
            print("=" * 80)
            start_time = time.time()
            
            # Extract query and context (parsed by pre-processing step)
            raw_query = user_input["raw_query"]
            query = user_input.get("processed_query", "")
            attachments = user_input.get("attachments", [])
            # print(user_input)

            # Process attachments if provided
            if attachments:
                attachment_contents = [att.get("content", "") for att in attachments if att.get("content")]
                image_paths = [att.get("path", "") for att in attachments if att.get("type") == "image" and att.get("path")]
            else:
                attachment_contents = []
                image_paths = []
           
            print("\n📥 **Input Details:**")
            print(f"  - Raw Query: {raw_query}")
            print(f"  - Number of Attachments: {len(attachments)}")
            if attachments:
                print("  - Attachment Details:")
                for i, att in enumerate(attachments, 1):
                    print(f"    {i}. Path: {att.get('path', 'N/A')}, Type: {att.get('type', 'N/A')}")

            # Step 1: Perform query analysis
            print("\n🔍 **Step 1: Query Analysis**")
            analysis_start = time.time()
            analysis_results = self.query_analyzer.analyze(query=raw_query, attachment_contents=attachment_contents)
            print(f"  - Rewritten Query: {analysis_results['rewritten_query']}")
            print(f"  - Keywords: {', '.join(analysis_results['keywords'])}")
            print(f"  - Time Related: {', '.join(analysis_results['time_related']) if analysis_results['time_related'] else 'None'}")
            print(f"  - Domain Area: {analysis_results['domain_area']}")

            print("\n⏱️ **Processing Time:**")
            print(f"  - Query Analysis Time: {time.time() - analysis_start:.2f}s")
            print("=" * 80)

            # Step 2: Route to appropriate retrievers
            print("\n🔀 **Step 2: Routing**")
            routing_start = time.time()
            routing_results = self.router.route(analysis_results)
            routing_results["selected_tools"].append("web_search")  # Always include web search as a fallback
            print(f"  - Routing to: {', '.join(routing_results['selected_tools'])} retriever(s)")
            print(f"  - Reasoning: {routing_results['reasoning']}")
            print(f"  - Retriever Metadata: {routing_results['retrieval_metadata']}")

            print("\n⏱️ **Processing Time:**")
            print(f"  - Routing Analysis Time: {time.time() - routing_start:.2f}s")
            print("=" * 80)

            # Step 3 & 4: Call the selected retriever to get documents and rerank them
            print("\n📚 **Step 3 & 4: Retrieval & Reranking**")
            retrieval_start = time.time()
            reranked_retrieval_results = self._retrieve_documents(routing_results, analysis_results)
            print(f"  - Reranked to final top {len(reranked_retrieval_results.contexts)} documents")

            print("\n⏱️ **Processing Time:**")
            print(f"  - Retrieval & Reranking Time: {time.time() - retrieval_start:.2f}s")
            print("=" * 80)

            # Step 5: Synthesize final answer with rewritten query and reranked contexts
            synthesis_start = time.time()
            final_answer = self.synthesizer.synthesize(
                raw_query=raw_query,
                query=analysis_results['rewritten_query'],
                rerank_result=reranked_retrieval_results,
            )

            print("\n✏️ **Step 5: Answer Synthesis**")
            print(f"  - Synthesis Time: {time.time() - synthesis_start:.2f}s")

            # Convert contexts to CLI-friendly sources list
            sources = final_answer.to_sources()

            print(f"✏️ Generated answer with {len(sources)} sources")
            print(f"⏱️ **Total Time:** {time.time() - start_time:.2f} seconds")
            print("=" * 80)

            return {
                "answer": final_answer.answer,
                "sources": sources,
                "confidence": 1.0,
                "analysis_results": analysis_results,
                "llm_metadata": final_answer.metadata,
            }

            # return {
            #     'answer': final_answer['answer'],
            #     'sources': final_answer['sources'],
            #     'confidence': final_answer.get('confidence', 0.0),
            #     'routing_info': routing_results,
            #     'retrieval_count': len(retrieval_results),
            #     'reranked_count': len(reranked_results)
            # }
    
        except Exception as e:
            print(f"❌ Error during query processing: {e}")
            return {
                'answer': f"Sorry, an error occurred while processing your query: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'error': str(e)
            }

    async def health_check(self) -> Dict[str, bool]:
        """Check health status of each component"""
        try:
            return {
                'query_analyzer': await self.query_analyzer.health_check(),
                'router': await self.router.health_check(),
                # Comment out this line since retrieval_manager is not initialized
                # 'retrieval_manager': await self.retrieval_manager.health_check(),
                'reranker': await self.reranker.health_check(),
                'synthesizer': await self.synthesizer.health_check()
            }
        except Exception as e:
            print(f"Health check error: {e}")
            return {
                'query_analyzer': False,
                'router': False,
                'retrieval_manager': False,  # You can keep this in the error return
                'reranker': False,
                'synthesizer': False
            }