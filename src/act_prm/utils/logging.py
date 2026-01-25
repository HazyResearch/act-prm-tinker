"""
Basic logging helpers
"""

import logging
from typing import Any

from omegaconf import OmegaConf, DictConfig, ListConfig
from rich import print as rich_print
from rich.errors import MarkupError
from rich.syntax import Syntax
from rich.tree import Tree


logger = logging.getLogger(__name__)


def print_header(x, border="both") -> None:
    """
    Print with borders
    """
    match border:
        case "both":
            prefix = f"{"-" * len(x)}\n"
            suffix = f"\n{"-" * len(x)}"
        case "top":
            prefix = f"{"-" * len(x)}\n"
            suffix = ""
        case "bottom":
            prefix = ""
            suffix = f"\n{"-" * len(x)}"
        case _:
            raise ValueError(f"Invalid border: {border}")
    rich_print(f"{prefix}{x}{suffix}")


def print_config(config: DictConfig, name: str = "CONFIG", style="bright") -> None:
    """
    Prints content of DictConfig using Rich library and its tree structure.
    """
    tree = Tree(name, style=style, guide_style=style)
    fields = config.keys()
    for field in fields:
        try:
            branch = tree.add(str(field), style=style, guide_style=style)
            config_section = config.get(field)
            branch_content = str(config_section)
            if isinstance(config_section, DictConfig):
                branch_content = OmegaConf.to_yaml(config_section, resolve=True)
            elif isinstance(config_section, ListConfig):
                branch_content = OmegaConf.to_yaml(config_section, resolve=True)
            branch.add(Syntax(branch_content, "yaml"))
        
        # except InterpolationResolutionError as e:
        except Exception as e:
            _error_text = f"({type(e).__name__}: {e})"
            print(f"-> Error resolving interpolation: {_error_text}")
            print(f"-> Field: {field}")
            print(f"-> Config section: {config_section}")
        
    rich_print(tree)


def rich_print_messages(
    msg_text: str,
    bos_token: str = "<|im_start|>",
    eos_token: str = "<|im_end|>\n",
    tool_call_bos_token: str = "<tool_call>",
    tool_call_eos_token: str = "</tool_call>",
    tool_response_bos_token: str = "<tool_response>",
    tool_response_eos_token: str = "</tool_response>",
    # Silly coloring
    system_color: str = "bright_yellow",
    user_color: str = "bright_red",
    assistant_color: str = "bright_cyan",
    tool_call_color: str = "dodger_blue1",
    tool_response_color: str = "bright_magenta",
    **rich_print_kwargs: Any,
) -> None:
    """
    Print chat-templated messages in silly colors
    """
    # Split into messages
    msgs = msg_text.split(eos_token)

    system_bos = f"{bos_token}system"
    user_bos = f"{bos_token}user"
    assistant_bos = f"{bos_token}assistant"
    
    for ix, msg in enumerate(msgs):
        # system prompt
        if msg.startswith(system_bos):
            msgs[ix] = f"[{system_color}]{msg}[/{system_color}]"
        # user messages
        elif msg.startswith(user_bos):
            msgs[ix] = f"[{user_color}]{msg}[/{user_color}]"
        # assistant messages
        elif msg.startswith(assistant_bos):
            msgs[ix] = f"[{assistant_color}]{msg}[/{assistant_color}]"
        
        # tool calls
        if tool_call_bos_token in msgs[ix] and tool_call_eos_token in msgs[ix]:
            msgs[ix] = msgs[ix].replace(tool_call_bos_token, f"[{tool_call_color}]{tool_call_bos_token}")
            msgs[ix] = msgs[ix].replace(tool_call_eos_token, f"{tool_call_eos_token}[/{tool_call_color}]")
        # tool responses
        if tool_response_bos_token in msgs[ix] and tool_response_eos_token in msgs[ix]:
            msgs[ix] = msgs[ix].replace(tool_response_bos_token, f"[{tool_response_color}]{tool_response_bos_token}")
            msgs[ix] = msgs[ix].replace(tool_response_eos_token, f"{tool_response_eos_token}[/{tool_response_color}]")
        
    msgs_text = eos_token.join(msgs)
    try:
        rich_print(msgs_text, **rich_print_kwargs)
    except MarkupError as e:
        logger.error(f"rich.errors.MarkupError: {e}")
        print(msgs_text)
