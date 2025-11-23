#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Analyzer Module
Responsible for analyzing user queries to enhance retrieval performance
"""

from typing import Dict, Any, List, Optional
import dateutil.parser as dparser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_openai import AzureChatOpenAI
# from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from ..prompts.templates import PromptTemplates
from config import get_settings
import datetime

settings = get_settings()

# Initialize templates
templates = PromptTemplates()

# Azure OpenAI client initialization
llm = AzureChatOpenAI(
    azure_endpoint=settings.azure_url,
    api_key=settings.azure_api_key,
    api_version="2025-02-01-preview",
)

class QueryOutput(BaseModel):
    """Schema for query analysis output"""
    keywords: List[str] = Field(description="Key terms extracted from the query")
    time_related: List[str] = Field(description="Time-related keywords extracted from the query (e.g., today, next Monday)")
    domain_area: str = Field(description="The Most relevant domain area for this query (e.g., finance, typhone, weather, transport, general)")

# Helper function to parse relative dates
def parse_time_info(times: List[str]) -> List[Optional[str]]:
    """
    Convert a list of relative date expressions (e.g., 'next Monday') to specific dates.

    Args:
        times: List of relative date expressions.

    Returns:
        A list of strings representing the exact dates in ISO format (YYYY-MM-DD), or None if parsing fails.
    """
    parsed_dates = []
    for time in times:
        try:
            if time.lower() in ["today", "tonight", "this evening", "this afternoon", "this morning"]:
                parsed_date = datetime.datetime.now()
            else:
                parsed_date = dparser.parse(time, fuzzy=True)
            formatted_date = parsed_date.strftime("%Y-%m-%d")
            parsed_dates.append(formatted_date)
        except Exception:
            parsed_dates.append(None)
    return parsed_dates
        

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
            
            self.rewrite_chain = self.rewrite_template | llm | StrOutputParser()

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
            rewritten_query = self.rewrite_chain.invoke(
                {
                "query": query,
                "context": "\n".join(attachment_contents) if attachment_contents else ""
                }
            )
            return rewritten_query
            
        except Exception as e:
            print(f"ERROR REWRITING QUERY: {e}")
            return query


class IntentExtractor:
    """Extracts and analyzes user intent from queries, including retriever metadata."""
    
    def __init__(self):
        """Initialize the intent extractor with local LLM."""
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=QueryOutput)
        
        try:
            # Fix the template category and name to match what's in templates.py
            template_content = templates.get_template('query_analysis', 'intent_extraction')
            
            # Create prompt template
            self.analysis_template = PromptTemplate(
                input_variables=["query", "format_instructions", "context"],
                template=template_content
            )
            self.analysis_template = self.analysis_template.partial(
                format_instructions=self.parser.get_format_instructions()
            )   # add format instructions based on the parser
            
            # Set up the chain
            self.analysis_chain = self.analysis_template | llm | self.parser

        except ValueError as e:
            raise ValueError(f"Missing required template: query_analysis.intent_extraction. Please add this template to templates.py: {e}")
            

    def analyze(self, query: str, attachment_contents: list = None) -> Dict[str, Any]:
        """
        Analyze user query to enhance retrieval
        
        Args:
            query: User's query text
            attachment_contents: List of attachment contents (optional)

        Returns:
        Query analysis metadata.
        """
        try:
            # Run analysis chain
            analysis_result = self.analysis_chain.invoke(
                {
                    "query": query,
                    "context": "\n".join(attachment_contents) if attachment_contents else ""
                }
            )

            # Parse the time-related keywords into exact dates or None
            analysis_result.time_related = parse_time_info(analysis_result.time_related)

            # Return enhanced query and metadata
            return {
                "original_query": '',
                "rewritten_query": query,
                "keywords": analysis_result.keywords,
                "time_related": [date for date in analysis_result.time_related if date is not None],  # Filter out None values
                "domain_area": analysis_result.domain_area,
            }

        except Exception as e:
            print(f"ERROR INTENT EXTRACTION: {e}")
            # Return original query if analysis fails
            return {
                "original_query": '',
                "rewritten_query": '',
                "keywords": [],
                "time_related": [],
                "domain_area": [],
            }
    

class QueryAnalyzer:
    """Orchestrates the query analysis pipeline by combining rewriting and intent extraction"""
    
    def __init__(self):
        """Initialize the query analyzer with rewriter and intent extractor"""
        self.query_rewriter = QueryRewriter()
        self.intent_extractor = IntentExtractor()

    def analyze(self, query: str, attachment_contents:list, use_attachments=True) -> Dict[str, Any]:
        """
        Complete query analysis pipeline: rewriting, intent analysis.
        
        Args:
            query: Original user query
            attachment_contents: List of attachment contents (optional)
            use_attachments: Flag indicating whether to use attachments in analysis

        Returns:
            Rewritten query and analysis metadata  
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
        analysis_results["rewritten_query"] = rewritten_query.strip('"')

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
