"""
Helper functions for processing Act-PRM samples
"""

from copy import copy
from functools import partial
from typing import Any

from datasets import Dataset


def get_thought_and_actions(
    msg: dict[str, Any],
    action_bos: str = "<tool_call>",
    action_eos: str = "</tool_call>",
    final_answer_bos: str = "Final Answer: ",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    From a given chat message, return two messages where the first only contains "thought text"
    and the second only contains "actions text".

    For example, if `msg` is of the form:
    {"role": "assistant", "content": "(thought text...)<tool_call>(actions text...)</tool_call>"}

    Then we return two message dicts:
    - {"role": "assistant", "content": "(thought text...)"}
    - {"role": "assistant", "content": "<tool_call>(actions text...)</tool_call>"}
    """
    thought_msg, actions_msg = msg, msg

    if msg["role"] == "assistant":
        thought_msg, actions_msg = copy(msg), copy(msg)
        content = msg["content"]
        if final_answer_bos is not None and final_answer_bos in content:
            thought_text, act_text = content.split(final_answer_bos)
            thought_msg["content"] = f"{thought_text.strip()}"
            actions_msg["content"] = f"{final_answer_bos}{act_text.strip()}"

        else:
            # Extract content from tool calls, e.g., f"<tool_call>{content}</tool_call>"
            assert action_bos in content, f"No explicit action found in message: {content}"
            thought_text, act_text = content.split(action_bos, maxsplit=1)
            assert action_eos in act_text, f"No closing action tag found in message: {act_text}"
            # Extract the action text content
            act_text = act_text.split(action_eos, maxsplit=1)[0].strip()
            # Add back the action tags (maybe redundant)
            act_text = f"{action_bos}{act_text}{action_eos}"
            thought_msg["content"] = f"{thought_text.strip()}"
            actions_msg["content"] = f"{act_text}"

    return thought_msg, actions_msg


def get_latent_completion(
    messages: list[dict[str, Any]],
    continue_final_message: bool = True,
    **get_thought_actions_kwargs: Any,
) -> tuple[list[dict[str, Any]], str]:
    """
    From standard chat messages, return tuple of (latent_inputs, action_target)
    where latent_inputs is a list of chat messages of the form:
    [
        {"role": "user", "content": <user_message>},
        {"role": "assistant", "content": "<action_message_0><thought_message_0>"},
        ...,
        {"role": "assistant", "content": "<action_message_last><thought_message_last>"},
    ]
    
    For training convenience, if `continue_final_message` is True, then we return `action_target`
    as the last assistant action message, e.g., "<action_message_last>"
    - We also change messages to prompt for the last thought, e.g., "<thought_message_last>"
    """
    latent_inputs = []
    action_target = ""

    assistant_indices = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] != "assistant":
            latent_inputs.append(msg)
        else:
            assistant_indices.append(msg_idx)
            msg_thought, msg_actions = get_thought_and_actions(msg, **get_thought_actions_kwargs)
            # Extract content from message dicts
            c_thought, c_actions = msg_thought["content"], msg_actions["content"]
            latent_content = f"{c_actions}\n\n<thought>\n{c_thought}\n</thought>"
            latent_content = latent_content.replace("\n\n\n", "\n\n")  # tidy a bit
            msg_latent = {"role": msg["role"], "content": latent_content}
            latent_inputs.append(msg_latent)

    if continue_final_message:
        # Make the last assistant message prompt for the latent thought
        last_assistant_idx = assistant_indices[-1]
        last_assistant_msg = latent_inputs[last_assistant_idx]
        msg_content = last_assistant_msg["content"].split("<thought>\n")[0]
        action_target = copy(msg_content).strip()
        msg_content = f"{msg_content}<thought>\n"
        latent_inputs[last_assistant_idx] = {
            "role": last_assistant_msg["role"],
            "content": msg_content,
        }
        latent_inputs = latent_inputs[:last_assistant_idx + 1]
    return latent_inputs, action_target


def filter_ds(example: dict[str, Any], **filter_kwargs: Any) -> bool:
    """
    Filter Dataset example
    """
    for k, v in filter_kwargs.items():
        if example[k] != v:
            return False
    return True


def get_ds_trajectory(ds_rb: Dataset, **filter_kwargs: Any) -> Dataset | None:
    """
    Select the samples in a Dataset that correspond to a single trajectory,
    filtering by the filter_kwargs (e.g., data_sample_id=0)
    """
    return ds_rb.filter(partial(filter_ds, **filter_kwargs))


def get_chat_from_ds_sample(
    row: dict[str, Any],
    column_names: list[str] | None = None,
) -> tuple[list[dict[str, str]], tuple[float, float]]:
    """
    Build list[dict[str, str]] message chat from Dataset sample
    """
    column_names = column_names or ["state", "action", "next_obs"]

    system_prompt: dict = row["system_prompt"]
    state: list[dict] = 
    action: dict = row["action"]
    next_obs: dict = row["next_obs"]
    next_obs["content"] = next_obs["content"].split("\n\nextracted_final_answer:")[0]
    reward:  float = row["reward"]
    return_: float = row["return_"]

    chat = [system_prompt]
    if "state" in column_names:
        chat += list(state)
    if "action" in column_names:
        chat += [action]
    if "next_obs" in column_names:
        chat += [next_obs]
    return chat, (reward, return_)
