#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt模板库
集中管理所有LLM交互的提示词模板
"""

from typing import Dict, List, Any, Optional


class PromptTemplates:
    """Prompt Templates Manager"""

    def __init__(self):
        """Initialize with predefined templates."""
        self.templates = {
            # Query analysis templates
            'query_analysis': {
                'intent_extraction': """
Please analyze the following user query to extract intent, keywords, and metadata.

User Query: {query}

Provide the following information:
1. Enhanced version of the query that clarifies intent
2. Key search terms/keywords (as a list)
3. Query type (factual, conceptual, procedural, opinion)
4. Domain areas relevant to the query (as a list)
5. Complexity level (easy, medium, hard)

{format_instructions}
""",

                'query_rewriting': """
Rewrite the following query to make it more effective for information retrieval.
Maintain the original intent but make it more specific, clear, and search-friendly.

Original query: {query}
Additional context: {context}

Rewritten query:
"""
            },

            # Routing templates
            'routing': {
                'tool_selection': """
You are an intelligent router that selects the most appropriate tools for a given user query.
User Query: {query}
Enhanced Query: {enhanced_query}
Available Tools: {tools}
Based on the query, select up to 3 tools that are most relevant. Provide a brief justification for each selected tool.
Selected Tools:
""",    
            },
            
            # 答案合成模板
            'synthesis': {
                'basic_qa': """
请基于以下检索到的信息回答用户问题。

用户问题: {query}

相关信息:
{documents}

回答要求:
1. 基于提供的信息准确回答
2. 如果信息不足，请明确说明
3. 保持回答简洁清晰
4. 标注信息来源

回答:
""",

                'detailed_analysis': """
请对以下问题进行详细分析和回答：

问题: {query}
上下文: {context}

参考资料:
{documents}

请提供:
1. 问题分析
2. 详细回答
3. 相关建议
4. 信息来源

分析和回答:
""",

                'comparison': """
请比较和分析以下信息，回答用户问题：

问题: {query}

待比较的信息:
{documents}

请提供:
1. 各项信息的对比分析
2. 优缺点总结
3. 推荐建议
4. 依据说明

比较分析:
"""
            },

            # 检索相关模板
            'retrieval': {
                'query_expansion': """
请扩展以下查询，生成更多相关的搜索关键词：

原始查询: {query}

请生成5-10个相关的搜索关键词或短语，用于提高检索效果。

扩展关键词:
""",

                'query_clarification': """
用户查询可能存在歧义，请帮助澄清：

查询: {query}
上下文: {context}

可能的理解:
1. ...
2. ...
3. ...

最可能的意图:
"""
            },

            # 错误处理模板
            'error_handling': {
                'no_results': """
抱歉，我没有找到与您问题相关的信息。

您的问题: {query}

建议:
1. 尝试使用不同的关键词
2. 提供更多背景信息
3. 检查问题的表述是否清楚

请问您还有其他问题吗？
""",

                'insufficient_info': """
根据现有信息，我只能提供部分回答：

问题: {query}
可用信息: {available_info}

部分回答: {partial_answer}

为了提供更完整的回答，建议:
{suggestions}
"""
            }
        }

    def get_template(self, category: str, template_name: str) -> str:
        """获取指定模板"""
        if category not in self.templates:
            raise ValueError(f"模板类别不存在: {category}")

        if template_name not in self.templates[category]:
            raise ValueError(f"模板不存在: {category}.{template_name}")

        return self.templates[category][template_name]

    def format_template(self, category: str, template_name: str, **kwargs) -> str:
        """格式化模板"""
        template = self.get_template(category, template_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"模板参数缺失: {e}")

    def add_template(self, category: str, template_name: str, template_content: str):
        """添加新模板"""
        if category not in self.templates:
            self.templates[category] = {}

        self.templates[category][template_name] = template_content

    def list_templates(self) -> Dict[str, List[str]]:
        """列出所有可用模板"""
        return {category: list(templates.keys()) for category, templates in self.templates.items()}