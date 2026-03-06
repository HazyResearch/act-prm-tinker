"""
Tools for Snorkel Agent Finance Reasoning environment.

5 tools: get_descriptions, get_table_info, sql_query, calculator, respond_user
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..base import BaseTool

logger = logging.getLogger(__name__)


class GetDescriptionsTool(BaseTool):
    """Get list of available financial tables for a company."""

    def __call__(
        self, company_name: str, backend: Any, **kwargs: Any
    ) -> str:
        return backend.get_descriptions(company_name)

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "get_descriptions",
            "description": (
                "Get the list of available financial tables/documents for a company. "
                "Returns a JSON array of table names from the company's 10-K filing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The company name to query (e.g., 'at_t', 'meta', 'disney')",
                    },
                },
                "required": ["company_name"],
            },
        }


class GetTableInfoTool(BaseTool):
    """Get metadata about a specific financial table."""

    def __call__(
        self,
        company_name: str,
        table_name: str,
        backend: Any,
        **kwargs: Any,
    ) -> str:
        return backend.get_table_info(company_name, table_name)

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "get_table_info",
            "description": (
                "Get metadata about a financial table including column names, "
                "unique values per column, and column data types."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The company name to query",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "The table name (from get_descriptions results)",
                    },
                },
                "required": ["company_name", "table_name"],
            },
        }


class SqlQueryTool(BaseTool):
    """Execute SQL queries on financial tables."""

    def __call__(
        self,
        company_name: str,
        table_name: str,
        query: str,
        backend: Any,
        **kwargs: Any,
    ) -> str:
        return backend.sql_query(company_name, table_name, query)

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "sql_query",
            "description": (
                "Execute a SQL query on a financial table. "
                "Must include specific column names (no SELECT *) and WHERE filters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The company name to query",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "The table name to query",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "SQL SELECT query with WHERE filters. "
                            "No SELECT * allowed."
                        ),
                    },
                },
                "required": ["company_name", "table_name", "query"],
            },
        }


class CalculatorTool(BaseTool):
    """Execute Python math expressions."""

    def __call__(self, expression: str, **kwargs: Any) -> str:
        try:
            # Safe eval: only allow math operations
            allowed_names = {"__builtins__": {}}
            import math

            allowed_names.update(
                {k: v for k, v in vars(math).items() if not k.startswith("_")}
            )
            result = eval(expression, allowed_names)  # noqa: S307
            return json.dumps([result, f"Calculated: {expression} = {result}"])
        except Exception as e:
            return f"Calculator error: {type(e).__name__}: {e}"

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "calculator",
            "description": (
                "Execute a Python math expression and return the result. "
                "Supports standard math operations (+, -, *, /, **, %) "
                "and math module functions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python math expression (e.g., '(310 / 6759) * 100')",
                    },
                },
                "required": ["expression"],
            },
        }


class RespondUserTool(BaseTool):
    """Provide final answer to the user."""

    def __call__(self, text: str, **kwargs: Any) -> str:
        # This is handled by the environment's step logic (triggers grading)
        return text

    def get_tool_desc(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "respond_user",
            "description": (
                "Provide your final answer to the user's question. "
                "Generate a concise paragraph summarizing your findings "
                "with all relevant figures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Your final answer text",
                    },
                },
                "required": ["text"],
            },
        }


TOOL_CLASSES = {
    "get_descriptions": GetDescriptionsTool,
    "get_table_info": GetTableInfoTool,
    "sql_query": SqlQueryTool,
    "calculator": CalculatorTool,
    "respond_user": RespondUserTool,
}
