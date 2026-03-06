"""
Trace-replay backend for Snorkel Agent Finance Reasoning.

Extracts tool call → response mappings from the snorkelai/agent-finance-reasoning
dataset traces, building a lookup table for each (tool_name, args) combination.

For novel queries not in the traces, returns an informative error.
The calculator tool is always live (uses Python eval).

The HF dataset stores traces in LangChain serialization format. We convert
these to our internal format (consistent with ActionFromLLM / action_utils.py)
before extracting tool call → response pairs for caching.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from datasets import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _normalize_key(*args: str) -> str:
    """Create a normalized lookup key from arguments."""
    return "|".join(str(a).strip().lower() for a in args)


def _convert_langchain_trace(
    trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a LangChain-serialized trace to our internal message format.

    LangChain format:
    - AI messages: type="ai", content is a list with
      {"type": "tool_use", "name": ..., "input": ...} items
    - Tool responses: type="tool", content is a string

    Internal format (consistent with ActionFromLLM / action_utils.py):
    - Assistant messages with tool calls:
      {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": ..., "arguments": ...}}
      ]}
    - Assistant text messages:
      {"role": "assistant", "content": "..."}
    - Tool responses:
      {"role": "tool", "type": "function_call_output",
       "call_id": ..., "output": "..."}
    - User messages:
      {"role": "user", "content": "..."}
    """
    converted = []
    for msg in trace:
        msg_type = msg.get("type", msg.get("role", ""))
        content = msg.get("content", "")

        if msg_type in ("human", "user"):
            converted.append({"role": "user", "content": content})

        elif msg_type in ("ai", "assistant"):
            tool_calls = []
            text_parts = []

            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "tool_use":
                        # LangChain tool_use → our function call format
                        tool_calls.append({
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("input", {}),
                            },
                        })
                    elif item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
            elif isinstance(content, str):
                text_parts.append(content)

            if tool_calls:
                msg_out: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }
                if text_parts:
                    msg_out["content"] = "\n".join(text_parts)
                converted.append(msg_out)
            elif text_parts:
                converted.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts),
                })

        elif msg_type in ("tool", "function"):
            converted.append({
                "role": "tool",
                "type": "function_call_output",
                "call_id": msg.get("id"),
                "output": content,
            })

    return converted


class TraceReplayBackend:
    """
    Backend that replays tool responses from recorded traces.

    Builds lookup tables from the original dataset:
    - descriptions_cache: company_name → response
    - table_info_cache: (company_name, table_name) → response
    - sql_cache: (company_name, table_name, query) → response
    """

    def __init__(self, traces: list[list[dict[str, Any]]]) -> None:
        self.descriptions_cache: dict[str, str] = {}
        self.table_info_cache: dict[str, str] = {}
        self.sql_cache: dict[str, str] = {}

        self._build_caches(traces)
        logger.info(
            f"TraceReplayBackend: {len(self.descriptions_cache)} descriptions, "
            f"{len(self.table_info_cache)} table_info, "
            f"{len(self.sql_cache)} sql_query entries"
        )

    def _build_caches(self, traces: list[list[dict[str, Any]]]) -> None:
        """Parse all traces and extract tool call → response mappings."""
        for trace in tqdm(traces, desc="Building trace-replay caches"):
            self._parse_trace(trace)

    def _parse_trace(self, trace: list[dict[str, Any]]) -> None:
        """Extract tool calls and their responses from a single trace.

        Converts LangChain trace to internal format, then extracts
        (tool_name, args) → response pairs for caching.
        """
        converted = _convert_langchain_trace(trace)

        for i, msg in enumerate(converted):
            if msg.get("role") != "assistant":
                continue

            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                continue

            # Collect consecutive tool responses after this assistant message
            tool_responses = []
            for j in range(i + 1, len(converted)):
                next_msg = converted[j]
                if next_msg.get("role") == "tool":
                    tool_responses.append(next_msg.get("output", ""))
                else:
                    break

            # Match tool calls to responses and cache
            for tc_idx, tc in enumerate(tool_calls):
                if tc_idx >= len(tool_responses):
                    break
                response_content = tool_responses[tc_idx]
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                args = func.get("arguments", {})

                self._cache_tool_response(tool_name, args, response_content)

    def _cache_tool_response(
        self, tool_name: str, args: dict[str, Any], response: str
    ) -> None:
        """Cache a tool response based on tool type and arguments."""
        if tool_name == "get_descriptions":
            company = args.get("company_name", "")
            key = _normalize_key(company)
            self.descriptions_cache[key] = response

        elif tool_name == "get_table_info":
            company = args.get("company_name", "")
            table = args.get("table_name", "")
            key = _normalize_key(company, table)
            self.table_info_cache[key] = response

        elif tool_name == "sql_query":
            company = args.get("company_name", "")
            table = args.get("table_name", "")
            query = args.get("query", "")
            key = _normalize_key(company, table, query)
            self.sql_cache[key] = response

    def get_descriptions(self, company_name: str) -> str:
        key = _normalize_key(company_name)
        if key in self.descriptions_cache:
            return self.descriptions_cache[key]
        # Try fuzzy match
        for cached_key, value in self.descriptions_cache.items():
            if company_name.lower() in cached_key:
                return value
        return (
            f"No data available for company '{company_name}'. "
            f"Available companies: {self.list_companies()}"
        )

    def get_table_info(self, company_name: str, table_name: str) -> str:
        key = _normalize_key(company_name, table_name)
        if key in self.table_info_cache:
            return self.table_info_cache[key]
        return (
            f"No table info found for table '{table_name}' "
            f"in company '{company_name}'. "
            f"Try calling get_descriptions first to see available tables."
        )

    def sql_query(self, company_name: str, table_name: str, query: str) -> str:
        key = _normalize_key(company_name, table_name, query)
        if key in self.sql_cache:
            return self.sql_cache[key]
        # Try partial match on company + table (ignoring query differences)
        partial_key = _normalize_key(company_name, table_name)
        available_queries = [
            k.split("|", 2)[-1]
            for k in self.sql_cache
            if k.startswith(partial_key)
        ]
        if available_queries:
            return (
                "Query not found in recorded data. "
                "Try one of these recorded queries for this table:\n"
                + "\n".join(f"- {q}" for q in available_queries[:5])
            )
        return (
            f"No SQL data found for table '{table_name}' "
            f"in company '{company_name}'."
        )

    def list_companies(self) -> list[str]:
        """List all companies with cached descriptions."""
        return sorted(set(self.descriptions_cache.keys()))

    @classmethod
    def from_dataset(cls, ds: Dataset) -> TraceReplayBackend:
        """Build backend from the snorkelai/agent-finance-reasoning HF dataset."""
        traces = []
        for sample in tqdm(ds, desc="Parsing dataset traces"):
            trace = sample.get("trace", [])
            if isinstance(trace, str):
                try:
                    trace = json.loads(trace)
                except json.JSONDecodeError:
                    continue
            if isinstance(trace, list) and len(trace) > 0:
                traces.append(trace)
        return cls(traces)
