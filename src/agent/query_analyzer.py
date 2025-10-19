#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Analyzer Module
Responsible for analyzing user queries to enhance retrieval performance
"""

import os
import base64
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
    
    # def _encode_image(self, image_path: str) -> str:
    #     """Encode an image file to base64"""
    #     with open(image_path, "rb") as image_file:
    #         return base64.b64encode(image_file.read()).decode('utf-8')

    def rewrite(self, query: str, images: list = None) -> str:
        """
        Rewrite query to be more effective for analysis and retrieval, incorporating visual content
        
        Args:
            query: Original user query
            images: List of image file paths (optional)
            
        Returns:
            Rewritten query that incorporates information from images
        """
        try:
            rewritten = self.rewrite_chain.invoke({
                "query": query,
                "context": ""  # No additional context for now
            })
            rewritten_query = rewritten.get("text", query)

            return rewritten_query
        
            # # Text-only approach (use standard chain)
            # if not images or len(images) == 0:
            #     rewritten = self.rewrite_chain.run(
            #         query=query,
            #         has_images=False,
            #         context=""
            #     )
            #     return rewritten.strip()
                
            # # Multimodal approach (direct model call with images)
            # else:
            #     # Create content list for the vision model
            #     prompt_text = self.rewrite_template.format(
            #         query=query,
            #         has_images=True
            #     )
                
            #     content = [{"type": "text", "text": prompt_text}]
                
            #     # Add images
            #     for img_path in images:
            #         try:
            #             img_base64 = self._encode_image(img_path)
            #             content.append({
            #                 "type": "image_url",
            #                 "image_url": {
            #                     "url": f"data:image/jpeg;base64,{img_base64}"
            #                 }
            #             })
            #         except Exception as e:
            #             print(f"Error processing image {img_path}: {e}")
                
            #     # Make direct call to the model
            #     response = self.llm.invoke(content)
            #     return response.content.strip()
            
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
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze user query to enhance retrieval
        
        Args:
            query: User's query text 
            
        Returns:
            Enhanced query and analysis metadata
        """
        try:
            # Run analysis chain
            analysis_result = self.analysis_chain.invoke(
                {"query": query}
            )['text']
            
            # Return enhanced query and metadata
            return {
                "original_query": '',
                "rewritten_query": query,
                "keywords": analysis_result.keywords,
                "query_type": analysis_result.query_type,
                "domain_areas": analysis_result.domain_areas,
                "complexity": analysis_result.complexity
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
                "complexity": "medium"
            }
    

class QueryAnalyzer:
    """Orchestrates the query analysis pipeline by combining rewriting and intent extraction"""
    
    def __init__(self):
        """Initialize the query analyzer with rewriter and intent extractor"""
        self.query_rewriter = QueryRewriter()
        self.intent_extractor = IntentExtractor()

    def analyze(self, query: str, images:list) -> Dict[str, Any]:
        """
        Complete query analysis pipeline: rewriting followed by intent analysis
        
        Args:
            query: Original user query
            images: List of image file paths (optional)
            
        Returns:
            Enhanced query and analysis metadata
        """
        # Step 1: Rewrite the query for better quality
        print(f"\n📝 Original query: \"{query}\"")
        print(f"🔄 Rewriting query{' with images' if images else ''}...")
        rewritten_query = self.query_rewriter.rewrite(query=query, images=images)
        print(f"✨ Rewritten query: \"{rewritten_query}\"")
        
        # Step 2: Analyze the rewritten query for intent and metadata
        print(f"🧠 Extracting intent from rewritten query...")
        analysis_results = self.intent_extractor.analyze(rewritten_query)
        
        # Add original query to results
        analysis_results["original_query"] = query
        
        # Add rewritten query as an intermediate step
        analysis_results["rewritten_query"] = rewritten_query

        # Add multimodal info
        analysis_results["has_images"] = len(images) > 0

        # Print key analysis results
        print(f"🔍 Query analysis results:")
        print(f"  - Original query: \"{analysis_results['original_query']}\"")
        print(f"  - Rewritten query: \"{analysis_results['rewritten_query']}\"")
        print(f"  - Query type: {analysis_results['query_type']}")
        print(f"  - Complexity: {analysis_results['complexity']}")
        print(f"  - Keywords: {', '.join(analysis_results['keywords'][:5])}" + 
            (f"... (+{len(analysis_results['keywords']) - 5} more)" if len(analysis_results['keywords']) > 5 else ""))
        print(f"  - Domain areas: {', '.join(analysis_results['domain_areas'])}")

        return analysis_results
    