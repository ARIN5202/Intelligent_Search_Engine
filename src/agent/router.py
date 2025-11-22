#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Router - Task 2 / Step 2
Analyze user query and decide which retrieval tools to call
"""

import json

from typing import Dict, Any, Optional
from warnings import filters

from sentence_transformers import SentenceTransformer, util
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from ..prompts.templates import PromptTemplates
from ..retrieval.manager import retrieval_manager

from config import get_settings

settings = get_settings()

print(settings.azure_url, settings.azure_api_key)

llm = AzureChatOpenAI(
    azure_endpoint=settings.azure_url,
    api_key=settings.azure_api_key,
    api_version="2025-02-01-preview",
    deployment_name="gpt-4o-mini",
    temperature=0.3,
    max_tokens=1000,
) 

class RoutingOutput(BaseModel):
    """Schema for routing output"""
    selected_tool: Optional[str] = Field(
        description="The name of the selected tool. If no tool is selected, this will be null."
    )
    description: Optional[str] = Field(
        description="A brief description of the selected tool. If no tool is selected, this will be null."
    )
    reasoning: str = Field(description="Overall reasoning for the tool selection.")

class RetrievalMetadataOutput(BaseModel):
    """Schema for retrieval metadata output"""
    required_fields: Dict[str, Any] = Field(
        description="A dict including only the required metadata fields for the selected tool."
    )

class Router:
    """Smart Router that analyzes queries and selects appropriate retrieval tools"""

    def __init__(self):
        """Initialize Router"""
        self.retrieval_manager = retrieval_manager
        # self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = llm
        self.routing_parser = PydanticOutputParser(pydantic_object=RoutingOutput)
        self.retrieval_metadata_parser = PydanticOutputParser(pydantic_object=RetrievalMetadataOutput)

        # Initialize tool selection and retrieval metadata prompt templates
        try:
            template_content = PromptTemplates().get_template('routing', 'tool_selection')
            
            self.routing_template = PromptTemplate(
                input_variables=["rewritten_query", "keywords", "domain_areas", "tool_descriptions"],
                template=template_content
            )
            self.routing_template = self.routing_template.partial(
                format_instructions=self.routing_parser.get_format_instructions()
            )

            template_content = PromptTemplates().get_template('routing', 'retrieval_metadata')

            self.retrieval_metadata_template = PromptTemplate(
                input_variables=["tool_name", "query"],
                template=template_content
            )
            self.retrieval_metadata_template = self.retrieval_metadata_template.partial(
                format_instructions=self.retrieval_metadata_parser.get_format_instructions()
            )

            self.routing_chain = self.routing_template | self.llm | self.routing_parser
            self.retrieval_metadata_chain = self.retrieval_metadata_template | self.llm | self.retrieval_metadata_parser
            
        except ValueError as e:
            raise ValueError(f"Missing required template: routing.tool_selection or routing.retrieval_metadata. Please add this template to templates.py: {e}")

    def _get_available_tool_info(self) -> Dict[str, Dict[str, Any]]:
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

    def _extract_retrieval_metadata(self, retriever_name: str, query: str) -> Dict[str, Any]:
        """
        Extract the required information for the selected tool.

        Args:
            retriever_name: Name of the selected retriever
            query: The user query
        Returns:
            A dictionary containing the extracted retrieval metadata.
        """
        retrieval_metadata = {}
        try:    
            retrieval_metadata = self.retrieval_metadata_chain.invoke(
                {
                    "tool_name": retriever_name,
                    "query": query
                }
            )
            return{
                **retrieval_metadata.required_fields
            }
        except Exception as e:
            print(f"ERROR RETRIEVAL METADATA EXTRACTION: {e}")
            return {}
        
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
            tool_info = self._get_available_tool_info()

            # Extract the keywords from analysis results
            query = analysis_results.get("rewritten_query") or analysis_results.get("raw_query") or ""
            query_keywords = analysis_results.get("keywords", [])
            query_domains = analysis_results.get("domain_areas", [])

            # Initialize routing variables
            best_tool = None
            reasoning = ""
            retrieval_metadata = {}

            # 1) quick word matching based on domains
            for tool_name, info in tool_info.items():
                tool_domains = info.get("domains", [])
                # print(f"Checking tool '{tool_name}' with domains {tool_domains} against query domains {query_domains}")
                if any(domain.lower() in tool_domains for domain in query_domains):
                    best_tool = tool_name
                    reasoning = f"Exact domain match found for tool '{best_tool}'."

                    retrieval_metadata = self._extract_retrieval_metadata(
                        retriever_name=best_tool,
                        query=query
                    )
                    break

            # # 2) semantic matching with embeddings if no exact match
            # if not best_tool:
            #     q_emb = self.model.encode(query, convert_to_tensor=True)
            #     scores = {}
            #     for tool_name, info in tool_info.items():
            #         tool_emb = self.model.encode(" ".join(info.get("domains", [])), convert_to_tensor=True)
            #         sim = util.cos_sim(q_emb, tool_emb).item()
            #         scores[tool_name] = sim

            #     # Select tools with similarity above threshold
            #     sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            #     top_tool, top_score = sorted_tools[0] if sorted_tools else (None, 0.0)
            #     if top_score >= 0.5:  # similarity threshold
            #         best_tool = top_tool
            #         reasoning = f"Semantic match found for tool '{best_tool}' with similarity {top_score:.2f}."

            # 3) fallback to LLM classification if still ambiguous
            if not best_tool:
                tool_descriptions = ", ".join([f"{name}: {info['description']}" for name, info in tool_info.items()])

                routing_result = self.routing_chain.invoke(
                    {
                        "rewritten_query": query,
                        "tool_descriptions": tool_descriptions,
                        "keywords": ", ".join(query_keywords),
                        "domain_areas": ", ".join(query_domains),
                    }
                )

                # Validate the routing results
                if routing_result.selected_tool:
                    best_tool = routing_result.selected_tool
                    retrieval_metadata = self._extract_retrieval_metadata(
                        retriever_name=best_tool,
                        query=query
                    )
                    reasoning = routing_result.reasoning
            
            if not best_tool:
                best_tool = "web_search"
                reasoning = "No suitable tool found; defaulting to 'web_search'."
            
            return {
                "selected_tool": best_tool,
                "reasoning": reasoning,
                "retrieval_metadata": retrieval_metadata
            }

        except Exception as e:
            return {
                "selected_tool": "web_search",
                # "confidence_scores": {"local_rag": 0.0},
                # "top_k": 10,
                "reasoning": f"Error in routing: {e}. Defaulting to 'web_search'.",
                "retrieval_metadata": {}
            }

    def _determine_top_k(self, query: str) -> int:
        """Determine number of documents to retrieve based on query complexity"""
        if len(query.split()) > 10:
            return 15  # Complex queries require more information
        return 10

    async def health_check(self) -> bool:
        """Health check"""
        return True
