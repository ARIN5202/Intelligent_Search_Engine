#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Analyzer Module
Responsible for analyzing user queries to enhance retrieval performance
"""

from typing import Dict, Any, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from ..prompts.templates import PromptTemplates

# Initialize templates
templates = PromptTemplates()

# Azure OpenAI client initialization
llm = AzureChatOpenAI(
    api_version="2025-02-01-preview",
    deployment_name="gpt-4o-mini",
    temperature=0.3,
    max_tokens=1000,
)


class QueryAnalysis(BaseModel):
    """Schema for query analysis output"""
    keywords: List[str] = Field(description="Key terms extracted from the query")
    query_type: str = Field(description="Type of query (e.g., factual, conceptual, procedural)")
    domain_areas: List[str] = Field(description="Relevant domain areas for this query (e.g., healthcare, finance)")
    complexity: str = Field(description="Query complexity (easy, medium, hard)")
    intent: str = Field(description="User's intent (e.g., retrieve information, perform action)")


class QueryRewriter:
    """Component to rewrite queries for better analysis and retrieval"""

    def __init__(self):
        """Initialize the multimodal query analyzer with OpenAI's model"""
        self.llm = llm
        
        try:
            template_content = templates.get_template('query_analysis', 'query_rewriting')
            
            self.rewrite_template = PromptTemplate(
                input_variables=["query"],
                template=template_content
            )
            
            self.rewrite_chain = LLMChain(
                llm=self.llm,
                prompt=self.rewrite_template
            )

        except ValueError as e:
            raise ValueError(f"Missing required template: analysis.query_rewriting. Please add this template to templates.py: {e}")

    def rewrite(self, query: str, attachment_contents: list = None) -> str:
        """
        Rewrite query to be more effective for analysis and retrieval, incorporating visual content
        
        Args:
            query: Original user query
            attachment_contents: List of attachment contents parsed from pre-processing step
            
        Returns:
            Rewritten query that incorporates textual information from attachments
        """
        try:
            rewritten = self.rewrite_chain.invoke(
                {
                "query": query,
                "context": "\n".join(attachment_contents) if attachment_contents else ""
                }
            )
            rewritten_query = rewritten.get("text", query)

            return rewritten_query
            
        except Exception as e:
            print(f"ERROR REWRITING QUERY: {e}")
            return query


class IntentExtractor:
    """Extracts and analyzes user intent from queries"""
    
    def __init__(self):
        """Initialize the intent extractor with local LLM"""
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=QueryAnalysis)
        
        try:
            # Fix the template category and name to match what's in templates.py
            template_content = templates.get_template('query_analysis', 'intent_extraction')
            
            # Create prompt template
            self.analysis_template = PromptTemplate(
                input_variables=["query", "format_instructions"],
                template=template_content
            )
            
            # Set up the chain
            self.analysis_chain = LLMChain(
                llm=self.llm,
                prompt=self.analysis_template.partial(format_instructions=self.parser.get_format_instructions()),
                output_parser=self.parser
            )
        except ValueError as e:
            raise ValueError(f"Missing required template: query_analysis.intent_extraction. Please add this template to templates.py: {e}")

    def analyze(self, query: str, attachment_contents: list = None) -> Dict[str, Any]:
        """
        Analyze user query to enhance retrieval
        
        Args:
            query: User's query text
            attachment_contents: List of attachment contents (optional)

        Returns:
            Enhanced query and analysis metadata
        """
        try:
            # Run analysis chain
            analysis_result = self.analysis_chain.invoke(
                {"query": query,
                 "context": "\n".join(attachment_contents) if attachment_contents else ""   
                }
            )['text']
            
            # Return enhanced query and metadata
            return {
                "original_query": '',
                "rewritten_query": query,
                "keywords": analysis_result.keywords,
                "query_type": analysis_result.query_type,
                "domain_areas": analysis_result.domain_areas,
                "complexity": analysis_result.complexity,
                "intent": analysis_result.intent
            }
        except Exception as e:
            print(f"ERROR INTENT EXTRACTION: {e}")
            # Return original query if analysis fails
            return {
                "original_query": '',
                "rewritten_query": query,
                "keywords": [],
                "query_type": "unknown",
                "domain_areas": [],
                "complexity": "unknown",
                "intent": "unknown"
            }
    

class QueryAnalyzer:
    """Orchestrates the query analysis pipeline by combining rewriting and intent extraction"""
    
    def __init__(self):
        """Initialize the query analyzer with rewriter and intent extractor"""
        self.query_rewriter = QueryRewriter()
        self.intent_extractor = IntentExtractor()

    def analyze(self, query: str, attachment_contents:list, use_attachments=False) -> Dict[str, Any]:
        """
        Complete query analysis pipeline: rewriting followed by intent analysis
        
        Args:
            query: Original user query
            attachment_contents: List of attachment contents (optional)
            use_attachments: Flag indicating whether to use attachments in analysis

        Returns:
            Enhanced query and analysis metadata
        """
        # Step 1: Rewrite the query for better quality
        rewritten_query = self.query_rewriter.rewrite(
            query=query,
            attachment_contents=attachment_contents if use_attachments else None
        )
        
        # Step 2: Analyze the rewritten query for intent and metadata
        analysis_results = self.intent_extractor.analyze(
            rewritten_query,
            attachment_contents=attachment_contents if use_attachments else None
        )
        
        # Add original query to results
        analysis_results["original_query"] = query
        
        # Add rewritten query as an intermediate step
        analysis_results["rewritten_query"] = rewritten_query

        return analysis_results
    
    async def health_check(self) -> bool:
        """
        Perform a health check to ensure all components are initialized correctly.
        
        Returns:
            True if all components are healthy, False otherwise.
        """
        try:
            # Check if the rewriter and intent extractor are initialized
            if self.query_rewriter and self.intent_extractor:
                return True
        except Exception as e:
            print(f"Health check failed: {e}")
        return False
    