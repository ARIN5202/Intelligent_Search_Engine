#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Router - Task 2 / Step 2
Analyze user query and decide which retrieval tools to call
"""

from typing import Dict, Any
from warnings import filters

from sentence_transformers import SentenceTransformer, util
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from ..prompts.templates import PromptTemplates

from ..retrieval.manager import retrieval_manager


# Azure OpenAI client initialization
llm = AzureChatOpenAI(
    api_version="2025-02-01-preview",
    deployment_name="gpt-4o-mini",
    temperature=0.3,
    max_tokens=1000,
) 


class Router:
    """Smart Router that analyzes queries and selects appropriate retrieval tools"""

    def __init__(self):
        """Initialize Router"""
        self.retrieval_manager = retrieval_manager

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize Azure OpenAI LLM
        self.llm = llm
        
        # Initialize prompt templates
        try:
            template_content = PromptTemplates().get_template('routing', 'tool_selection')
            self.routing_template = PromptTemplate(
                input_variables=["query", "tool_descriptions", "keywords", "domain_areas", "intent"],
                template=template_content
            )
        except ValueError as e:
            raise ValueError(f"Missing required template: routing.tool_selection. Please add this template to templates.py: {e}")

    def get_available_tool_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch all available tools from the RetrievalManager.

        Returns:
            A dictionary where keys are tool names and values are tool metadata.
        """
        tool_info = {}
        for retriever_name in self.retrieval_manager.list_retrievers():
            retriever_instance = self.retrieval_manager.get_retriever(retriever_name)
            tool_info[retriever_name] = {
                "domains": getattr(retriever_instance, "domain", ["general"]),
                "description": getattr(retriever_instance, "description", "No description available.")
            }
        return tool_info

    def route(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the query and return routing results

        Args:
            analysis_results: Results from the query analysis step, including:
                - rewritten_query: The rewritten query string.
                - keywords: Extracted keywords from the query.
                - intent: The intent of the query.
                - domain_areas: The domain(s) the query belongs to.
                - complexity: The complexity of the query.

        Returns:
            A dictionary containing:
                - selected_tool: the most suitable retrieval tool(s) for the query
                - reasoning: Explanation of the routing decision
                - retriever_metadata: Additional metadata for the selected retriever
        """
        try: 
            # Fetch available tools dynamically
            tool_info = self.get_available_tool_info()

            # Extract the keywords from analysis results
            query = analysis_results.get("rewritten_query") or analysis_results.get("raw_query") or ""
            query_keywords = analysis_results.get("keywords", [])
            query_domains = analysis_results.get("domain_areas", [])
            # query_intent = analysis_results.get("intent", "").lower()

            # 1) quick word matching based on domains
            best_tool = None
            for tool_name, info in tool_info.items():
                tool_domains = info.get("domains", [])
                # print(f"Checking tool '{tool_name}' with domains {tool_domains} against query domains {query_domains}")
                if any(domain.lower() in tool_domains for domain in query_domains):
                    best_tool = tool_name
                    reasoning = f"Exact domain match found for tool '{best_tool}'."
                    break

            # 2) semantic matching with embeddings if no exact match
            if not best_tool:
                q_emb = self.model.encode(query, convert_to_tensor=True)
                scores = {}
                for tool_name, info in tool_info.items():
                    tool_emb = self.model.encode(" ".join(info.get("domains", [])), convert_to_tensor=True)
                    sim = util.cos_sim(q_emb, tool_emb).item()
                    scores[tool_name] = sim

                # Select tools with similarity above threshold
                sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_tool, top_score = sorted_tools[0] if sorted_tools else (None, 0.0)
                if top_score >= 0.5:  # similarity threshold
                    best_tool = top_tool
                    reasoning = f"Semantic match found for tool '{best_tool}' with similarity {top_score:.2f}."

            # 3) fallback to LLM classification if still ambiguous
            if not best_tool:
                try:
                    tool_descriptions = ", ".join([f"{name}: {info['description']}" for name, info in tool_info.items()])
                    # Use LLMChain to invoke the Azure OpenAI API
                    routing_chain = self.routing_template | self.llm | StrOutputParser()

                    response = routing_chain.invoke(
                        {
                            "query": query,
                            "tool_descriptions": tool_descriptions,
                            "keywords": ", ".join(query_keywords),
                            "domain_areas": ", ".join(query_domains),
                            # "intent": query_intent,
                        }
                    )

                    # Parse the response to extract the best tool
                    response_text = response.get("text", "").strip()
                    parsed_response = eval(response_text)  # Convert JSON-like string to Python dict

                    # Validate the parsed response
                    if "selected_tool" in parsed_response and "reasoning" in parsed_response:
                        best_tool = parsed_response["selected_tool"]["tool_name"]
                        reasoning = parsed_response["reasoning"]
                    else:
                        raise ValueError("Malformed response from LLM. Missing required fields.")
                    
                except Exception as e:
                    # Final fallback
                    best_tool = "web_search"
                    reasoning = f"Error in LLM-based routing: {e}. Defaulting to 'web_search'."

            return {
                "selected_tool": best_tool,
                # "confidence_scores": confidence if 'confidence' in locals() else {tool: 1.0 for tool in selected_tools},
                # "top_k": self._determine_top_k(query),
                "reasoning": reasoning
            }

        except Exception as e:
            return {
                "selected_tool": "web_search",
                # "confidence_scores": {"local_rag": 0.0},
                # "top_k": 10,
                "reasoning": f"Error in routing: {e}. Defaulting to 'web_search'."
            }

    def _determine_top_k(self, query: str) -> int:
        """Determine number of documents to retrieve based on query complexity"""
        if len(query.split()) > 10:
            return 15  # Complex queries require more information
        return 10

    async def health_check(self) -> bool:
        """Health check"""
        return True
