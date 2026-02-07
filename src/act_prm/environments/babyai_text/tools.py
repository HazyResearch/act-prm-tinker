"""
Tool-call mapping between BabyAI action strings and LLM function-call names.

BabyAI actions use natural-language names with spaces (e.g. ``"turn left"``),
but LLM tool calls use underscore-separated identifiers (e.g. ``"turn_left"``).
This module provides bidirectional mappings and a helper class that formats
tool calls for the LLM and converts them back to BabyAI actions.
"""

from __future__ import annotations

import json
from typing import Any

from ..base import BaseTool

ACTION_TO_TOOL = {
    "turn left": "turn_left",
    "turn right": "turn_right",
    "go forward": "go_forward",
    "pick up": "pick_up",
    "drop": "drop",
    "toggle": "toggle",
}

TOOL_TO_ACTION = {tool: action for action, tool in ACTION_TO_TOOL.items()}

# Tool descriptions in function-calling format with no params required.
TOOL_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "turn_left": {
        "type": "function",
        "name": "turn_left",
        "description": "Rotate the agent 90 degrees to the left.",
        "parameters": {"type": "object", "properties": {}},
    },
    "turn_right": {
        "type": "function",
        "name": "turn_right",
        "description": "Rotate the agent 90 degrees to the right.",
        "parameters": {"type": "object", "properties": {}},
    },
    "go_forward": {
        "type": "function",
        "name": "go_forward",
        "description": "Move forward one cell.",
        "parameters": {"type": "object", "properties": {}},
    },
    "pick_up": {
        "type": "function",
        "name": "pick_up",
        "description": "Pick up the object in front of the agent.",
        "parameters": {"type": "object", "properties": {}},
    },
    "drop": {
        "type": "function",
        "name": "drop",
        "description": "Drop the object the agent is carrying.",
        "parameters": {"type": "object", "properties": {}},
    },
    "toggle": {
        "type": "function",
        "name": "toggle",
        "description": "Toggle (use) the object in front of the agent.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class BabyAiTextTool(BaseTool):
    """
    Tool helper for converting between LLM tool-call names and BabyAI action strings.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the tool helper. Stores tool descriptions internally.

        Args:
            **kwargs: Passed through to ``BaseTool.__init__``.
        """
        super().__init__(**kwargs)
        self.tool_descriptions = TOOL_DESCRIPTIONS

    def __call__(self, tool_name: str) -> str:
        """
        Convert an LLM tool name to a BabyAI action string.

        Args:
            tool_name: The tool name from the LLM's function call
                (e.g. ``"go_forward"``).

        Returns:
            The corresponding BabyAI action string (e.g. ``"go forward"``),
            or ``""`` if the tool name is not recognized.
        """
        return TOOL_TO_ACTION.get(tool_name, "")

    def get_tool_descs(self) -> list[dict[str, Any]]:
        """
        Return all tool descriptions in function-calling format.

        Returns:
            A list of tool description dicts, one per BabyAI action.
        """
        return [TOOL_DESCRIPTIONS[name] for name in TOOL_TO_ACTION.keys()]

    def get_tool_desc(self, tool_name: str) -> dict[str, Any]:
        """
        Return the tool description for a specific tool.

        Args:
            tool_name: The LLM tool name (e.g. ``"go_forward"``).

        Returns:
            The tool description dict in OpenAI function-calling format.

        Raises:
            KeyError: If ``tool_name`` is not a valid tool.
        """
        return TOOL_DESCRIPTIONS[tool_name]

    def get_llm_toolcall_from_action(self, action_text: str, toolcall_tag: str = "tool_call") -> str:
        """
        Generate an XML-formatted tool-call string for gold demonstration trajectories.

        Args:
            action_text: The BabyAI action string (e.g. ``"go forward"``).
            toolcall_tag: The XML tag name to wrap the tool call in.
                Defaults to ``"tool_call"``.

        Returns:
            A string like::
                 <tool_call>
                {"name": "go_forward", "arguments": {}}
                 </tool_call> 
        """
        tool_name = ACTION_TO_TOOL.get(action_text, "")
        toolcall = {"name": tool_name, "arguments": {}}
        return f"<{toolcall_tag}>\n{json.dumps(toolcall)}\n</{toolcall_tag}>"