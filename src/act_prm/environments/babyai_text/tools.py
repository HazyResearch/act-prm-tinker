"""
Tools for BabyAI-Text environment
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
    Tool helper for mapping tool calls to BabyAI action strings
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tool_descriptions = TOOL_DESCRIPTIONS

    def __call__(self, tool_name: str) -> str:
        """
        Convert tool name to BabyAI action string
        """
        return TOOL_TO_ACTION.get(tool_name, "")

    def get_tool_descs(self) -> list[dict[str, Any]]:
        """
        Return all tool descriptions
        """
        return [TOOL_DESCRIPTIONS[name] for name in TOOL_TO_ACTION.keys()]

    def get_tool_desc(self, tool_name: str) -> dict[str, Any]:
        """
        Return tool description for a given tool name
        """
        return TOOL_DESCRIPTIONS[tool_name]

    def get_llm_toolcall_from_action(self, action_text: str, toolcall_tag: str = "tool_call") -> str:
        """
        Return a valid LLM tool call string from an action text
        """
        tool_name = ACTION_TO_TOOL.get(action_text, "")
        toolcall = {"name": tool_name, "arguments": {}}
        return f"<{toolcall_tag}>\n{json.dumps(toolcall)}\n</{toolcall_tag}>"
