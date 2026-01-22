"""
Base class for TextWorld tools
-> Basically lets us go from modern LLM function calling with tool descriptions, to the TextWorld command strings
"""

import json
from typing import Any

from act_prm.environments.base import BaseTool

ALLOWED_TASKS = ["coin_collector", "the_cooking_game", "treasure_hunter"]


class TextWorldTool(BaseTool):
    """
    TextWorld tool class
    
    For specific task-types (e.g., "coin_collector", "the_cooking_game", "treasure_hunter"),
    we load the tool descriptions and templates from the appropriate files
    """
    def __init__(self, task: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.task = task
        if task not in ALLOWED_TASKS:
            raise ValueError(f"Invalid task type: {task}. Allowed: {ALLOWED_TASKS}")
        self.tool_descriptions, self.tool_to_template = self.get_tool_descs_and_template()
        # Get mapping so we can convert TextWorld action text to LLM tool calls
        # -> For example: "take pepper from box" ->
        #    {"name": "take_from", "arguments": {"item": "pepper", "source": "box"}},
        #    instead of {"name": "take", "arguments": {"item": "pepper from box"}}
        self.tools_to_resolve_from_text = {} 
        for tool_name in self.tool_to_template.keys():
            if (
                len(tool_name.split("_")) > 1
                # and tool_name.split("_")[0] in self.tool_to_template.keys()
            ):
                # if a tool name has multiple words, use 2nd word's inclusion
                # in TextWorld action text to resolve the tool (only have max 2-word tools)
                _name_parts = tool_name.split("_")
                self.tools_to_resolve_from_text[_name_parts[0]] = _name_parts[1]

    def get_tool_descs_and_template(self, task: str | None = None) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """
        Returns a tuple of tool descriptions and templates
        """
        task = task or self.task
        
        if task == "coin_collector":
            from .coin_collector import TOOL_DESCRIPTIONS, TOOL_TO_TEMPLATE
        elif task == "the_cooking_game":
            from .cooking_game import TOOL_DESCRIPTIONS, TOOL_TO_TEMPLATE
        elif task == "treasure_hunter":
            from .treasure_hunter import TOOL_DESCRIPTIONS, TOOL_TO_TEMPLATE
        else:
            raise ValueError(f"Invalid task type: {task}. Allowed: {ALLOWED_TASKS}")
        
        return TOOL_DESCRIPTIONS, TOOL_TO_TEMPLATE

    def __call__(self, tool_name: str, tool_args: dict[str, str] | None) -> str:
        """
        Based on the provided tool name and arguments, returns a templated string
        that TextWorld environments can parse and execute
        """
        try:
            template = self.tool_to_template[tool_name]
            return template.format(**tool_args)
        except Exception as e:
            _error_class = type(e).__name__
            return f"Tool call error ({_error_class}: {e})"  # Pass this error message in env.py

    def get_tool_descs(self) -> list[dict[str, Any]]:
        """
        Returns a list of tool descriptions
        """
        return [self.tool_descriptions[tool_name] for tool_name in self.tool_to_template.keys()]

    def get_tool_desc(self, tool_name: str) -> dict[str, Any]:
        """
        Returns the tool description for a given tool name
        """
        return self.tool_descriptions[tool_name]

    def get_llm_toolcall_from_tw_text(self, tw_action: str, toolcall_tag: str = "tool_call") -> str:
        """
        Return the valid LLM tool call JSON from a given TextWorld action text
        -> Maybe Qwen3-coded for now

        For example:
        * 'go north' -> '<tool_call>\n{"name": "go", "arguments": {"direction": "north"}}\n</tool_call>'
        * 'prepare meal' -> '<tool_call>\n{"name": "prepare_meal", "arguments": {}}\n</tool_call>'
        * 'take white onion from fridge' -> '<tool_call>\n{"name": "take_from", "arguments": {"item": "white onion", "source": "fridge"}}\n</tool_call>'
        """
        # Resolve tool name clashes; e.g., to distinguish between "take" and "take_from"
        tool_name = tw_action.split(" ")[0]
        tie_break = self.tools_to_resolve_from_text.get(tool_name, "null")
        if tie_break != "null":  # should handle most cases
            new_tool_name = f"{tool_name}_{tie_break}" if tie_break in tw_action else tool_name
        else:
            new_tool_name = tool_name

        # Go through words in tw_action text to build the JSON tool call
        tw_arg_text = tw_action[len(tool_name):].strip()  # after the tool call
        tw_args = tw_arg_text.split(tie_break)

        toolcall_arguments = list(
            self.tool_descriptions[new_tool_name]["parameters"]["properties"].keys()
        )
        toolcall = {"name": new_tool_name, "arguments": {}}
        for arg_ptr, maybe_arg in enumerate(tw_args):
            if len(toolcall_arguments) > 0:
                toolcall["arguments"][toolcall_arguments[arg_ptr]] = maybe_arg.strip()

        return f"<{toolcall_tag}>\n{json.dumps(toolcall)}\n</{toolcall_tag}>"
