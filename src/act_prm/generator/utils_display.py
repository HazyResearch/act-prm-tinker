"""
Helper functions for displaying generator rollouts
"""

import logging
from typing import Any

from rich import print as rich_print
from rich.errors import MarkupError
from rich.panel import Panel
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def display_state_action_next_obs(
    state_messages: list[dict[str, Any]],
    action_messages: list[dict[str, Any]],
    next_obs_messages: list[dict[str, Any]],
    hf_tokenizer: PreTrainedTokenizerBase,
    tools: list[dict[str, Any]] | None = None,
    header_text: str = "Batch 0, Try 0, Sample 0, Generation 0, Step 0",
    panel_content: str | None = None,
    # silly coloring + formatting
    system_color: str = "bright_yellow",
    user_color: str = "bright_red",
    assistant_color: str = "bright_cyan",
    tool_call_color: str = "bright_blue",
    tool_call_bos: str = "<tool_call>",
    tool_call_eos: str = "</tool_call>",
    tool_response_color: str = "bright_cyan",
    tool_response_bos: str = "<tool_response>",
    tool_response_eos: str = "</tool_response>",
) -> None:
    """
    Display the state and action in a readable format
    """
    def add_role_marker(msg: dict[str, str], marker: str | None = None) -> dict[str, str]:
        """
        Add role marker to message content, so that we can style it after chat template formatting
        """
        role, content = msg["role"], msg["content"]
        if marker is not None:
            content = f"[{marker}]{content}"
        else:
            content = f"[{role}]{content}"
        return {"role": role, "content": content}

    state_messages = [add_role_marker(msg) for msg in state_messages]
    action_messages = [add_role_marker(msg, "last_action") for msg in action_messages]
    next_obs_messages = [add_role_marker(msg, "next_obs") for msg in next_obs_messages]

    all_text = hf_tokenizer.apply_chat_template(
        state_messages + action_messages + next_obs_messages,
        add_generation_prompt=False,
        tokenize=False,
        tools=tools,
    )
    messages = all_text.split(hf_tokenizer.eos_token)
    for ix, msg in enumerate(messages):
        if "[last_action]" in msg:
            msg = msg.replace("[last_action]", "")
            # messages[ix] = f"[bold bright_cyan]{msg}[/bold bright_cyan]"
            messages[ix] = f"[dim]{msg}[/dim]"
        elif "[next_obs]" in msg:
            msg = msg.replace("[next_obs]", "")
            messages[ix] = f"[dim italic]{msg}[/dim italic]"
        
        if "[system]" in msg:
            msg = msg.replace("[system]", "")
            messages[ix] = f"[{system_color}]{msg}[/{system_color}]"
        elif "[user]" in msg:
            msg = msg.replace("[user]", "")
            messages[ix] = f"[{user_color}]{msg}[/{user_color}]"
        elif "[tool]" in msg:
            msg = msg.replace("[tool]", "")
            messages[ix] = f"[{tool_response_color}]{msg}[/{tool_response_color}]"
        elif "[assistant]" in msg:
            msg = msg.replace("[assistant]", "")
            messages[ix] = f"[{assistant_color}]{msg}[/{assistant_color}]"

        # tool calls
        msg = messages[ix]
        if tool_call_bos in msg and tool_call_eos in msg:
            msg = msg.replace(tool_call_bos, f"[{tool_call_color}]{tool_call_bos}")
            msg = msg.replace(tool_call_eos, f"{tool_call_eos}[/{tool_call_color}]")
            messages[ix] = msg
        # tool responses
        if tool_response_bos in msg and tool_response_eos in msg:
            msg = msg.replace(tool_response_bos, f"[{tool_response_color}]{tool_response_bos}")
            msg = msg.replace(tool_response_eos, f"{tool_response_eos}[/{tool_response_color}]")
            messages[ix] = msg

    all_text = hf_tokenizer.eos_token.join(messages)
    try:
        rich_print(all_text)
        rich_print(Panel(panel_content, title=header_text, style="bold"))
    except MarkupError as e:
        header_text = f"{"-" * 50} {header_text} {"-" * 50}"
        print(f"{header_text}\n{all_text}\n{header_text}")
        rich_print(panel_content)
        logger.error(f"rich.errors.MarkupError: {e}")


def display_state_action_next_obs_old(
    state_messages: list[dict[str, Any]],
    action_messages: list[dict[str, Any]],
    next_obs_messages: list[dict[str, Any]],
    hf_tokenizer: PreTrainedTokenizerBase,
    tools: list[dict[str, Any]] | None = None,
    header_text: str = "Batch 0, Try 0, Sample 0, Generation 0, Step 0",
    panel_content: str | None = None,
    # silly coloring + formatting
    system_color: str = "bright yellow",
    user_color: str = "bright red",
    assistant_color: str = "bright cyan",
    tool_call_color: str = "bright blue",
    tool_call_bos: str = "<tool_call>",
    tool_call_eos: str = "</tool_call>",
    tool_response_color: str = "bright green",
    tool_response_bos: str = "<tool_response>",
    tool_response_eos: str = "</tool_response>",
) -> None:
    """
    Display the state and action in a readable format
    """

    def make_rich(msg: dict[str, str], style: str | None = None) -> dict[str, str]:
        """
        Apply silly coloring + formatting to content
        -> Hardcoded slightly to Qwen models for now
        """
        role, content = msg["role"], msg["content"]
        if style is not None:  # if style is provided, use it
            return {"role": role, "content": f"[{style}]{content}[/{style}]"}
        
        # Color system, user, and assistant messages
        if role == "system":
            content = f"[{system_color}]{content}[/{system_color}]"
        elif role == "user":
            content = f"[{user_color}]{content}[/{user_color}]"
        elif role == "assistant":
            content = f"[{assistant_color}]{content}[/{assistant_color}]"
        # Special color for tool calls
        if tool_call_bos in content:
            rich_tool_call_bos = f"[/{assistant_color}][{tool_call_color}]{tool_call_bos}"
            rich_tool_call_eos = f"{tool_call_eos}[/{tool_call_color}][{assistant_color}]"
            content = content.replace(tool_call_bos, rich_tool_call_bos)
            content = content.replace(tool_call_eos, rich_tool_call_eos)
        # Special color for tool responses
        # -> NOTE (MZ 1/11/2026): This doesn't get parsed before applying chat template
        if tool_response_bos in content:
            rich_tool_response_bos = f"[/{user_color}][{tool_response_color}]{tool_response_bos}"
            rich_tool_response_eos = f"{tool_response_eos}[/{tool_response_color}][{user_color}]"
            content = content.replace(tool_response_bos, rich_tool_response_bos)
            content = content.replace(tool_response_eos, rich_tool_response_eos)        
        return {"role": role, "content": content}
        
    state_messages = [make_rich(msg) for msg in state_messages]
    action_messages = [make_rich(msg, "bold bright_green") for msg in action_messages]
    next_obs_messages = [make_rich(msg, "bold bright_orange") for msg in next_obs_messages]
    
    all_text = hf_tokenizer.apply_chat_template(
        state_messages + action_messages + next_obs_messages,
        add_generation_prompt=False,
        tokenize=False,
        tools=tools,
    )
    # if tool_response_bos in all_text and tool_response_eos in all_text:
    #     rich_tool_response_bos = f"[{tool_response_color}]{tool_response_bos}"
    #     rich_tool_response_eos = f"{tool_response_eos}[/{tool_response_color}]"
    #     all_text = all_text.replace(tool_response_bos, rich_tool_response_bos)
    #     all_text = all_text.replace(tool_response_eos, rich_tool_response_eos)
    try:
        rich_print(all_text)
        rich_print(Panel(panel_content, title=header_text, style="bold"))
    except MarkupError as e:
        header_text = f"{"-" * 50} {header_text} {"-" * 50}"
        print(f"{header_text}\n{all_text}\n{header_text}")
        rich_print(panel_content)
        logger.error(f"rich.errors.MarkupError: {e}")
