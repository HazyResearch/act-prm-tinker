"""
Trace-replay backend for Snorkel Agent Finance Reasoning.

Extracts tool call → response mappings from the snorkelai/agent-finance-reasoning
dataset traces, building a lookup table for each (tool_name, args) combination.

For novel queries not in the traces, returns an informative error.
The calculator tool is always live (uses Python eval).
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
        """Extract tool calls and their responses from a single trace."""
        for i, msg in enumerate(trace):
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if "<tool_call>" not in content:
                continue

            # Extract tool call JSON
            try:
                tc_str = content.split("<tool_call>")[-1].split("</tool_call>")[0].strip()
                tc = json.loads(tc_str)
            except (json.JSONDecodeError, IndexError):
                continue

            tool_name = tc.get("name", "")
            args = tc.get("arguments", {})

            # Find the next tool response message
            response_content = None
            for j in range(i + 1, len(trace)):
                next_msg = trace[j]
                if next_msg.get("role") in ["tool", "function"]:
                    response_content = next_msg.get("content", next_msg.get("output", ""))
                    break
                if next_msg.get("role") == "assistant":
                    break  # no tool response found before next assistant msg

            if response_content is None:
                continue

            # Cache based on tool type
            if tool_name == "get_descriptions":
                company = args.get("company_name", "")
                key = _normalize_key(company)
                self.descriptions_cache[key] = response_content

            elif tool_name == "get_table_info":
                company = args.get("company_name", "")
                table = args.get("table_name", "")
                key = _normalize_key(company, table)
                self.table_info_cache[key] = response_content

            elif tool_name == "sql_query":
                company = args.get("company_name", "")
                table = args.get("table_name", "")
                query = args.get("query", "")
                key = _normalize_key(company, table, query)
                self.sql_cache[key] = response_content

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
