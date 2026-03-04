"""
Utils for provenance in setting up Tau(2)-Bench observational data

We use the great traces available at: https://huggingface.co/datasets/amityco/apigen-tau-bench-split-turn
"""

import ast
import json
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from tqdm.autonotebook import tqdm


# Tool we use for any direct messages from assistant to user
respond_user_tool = {
    "type": "function",
    "name": "respond_user",
    "description": "Respond or message the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text content to respond or message the user.",
            }
        },
        "required": ["text"],
    },
}


def convert_msg_to_think(
    message: dict[str, str],
    tool_call_bos: str = "<tool_call>",
    tool_call_eos: str = "</tool_call>",
) -> tuple[dict[str, str], list[dict[str, Any]], list[str]]:
    """
    Convert an assistant message to our "think-act" format
    """
    new_msg: dict[str, str] = {}
    new_tools: list[dict[str, Any]] = []
    old_texts: list[str] = []

    role = message["role"]
    if role != "assistant":  # Only apply to assistant messages
        message = {
            "role": role,
            "content": message["content"]
        }
        return message, new_tools, old_texts

    # Build the message content in think-act format
    _new_msg_contents: list[str] = []
    old_text = ""
    reasoning_content = message.get("reasoning_content", "") or ""
    content = message.get("content", "") or ""
    tool_calls = message.get("tool_calls", []) or []

    if len(reasoning_content) > 0:  # add thought inline
        _new_msg_contents.append(reasoning_content)
        old_text = reasoning_content

    assert not (len(content) > 0 and len(tool_calls) > 0), (
        "Cannot have both content and tool calls"
    )
    try:
        assert len(content) > 0 or len(tool_calls) > 0, (
            "Must have either content or tool calls"
        )
    except AssertionError as e:
        print(f"{e.__class__.__name__}: {e}")
        print(f"Message: {message}")
        print("-" * 100)
        return None, None, None
    
    if len(content) > 0:  # assistant is responding to user
        tool_dict = {
            "name": "respond_user",
            "arguments": {"text": content},
        }
        _new_msg_contents.append(f"{tool_call_bos}\n{json.dumps(tool_dict)}\n{tool_call_eos}")
        new_tools.append(tool_dict)
        old_texts.append(old_text)

    for tool_call in tool_calls:
        tool_call = tool_call["function"]
        try:
            tool_dict = {
                "name": tool_call["name"],
                "arguments": json.loads(tool_call["arguments"])
            }
            if isinstance(tool_dict["arguments"], str):
                print(f"Original json.loads(tool_call['arguments']): {tool_dict['arguments']}, (type: {type(tool_dict['arguments'])})")
                tool_dict["arguments"] = json.loads(tool_dict["arguments"])
                print(f"After 2nd json.loads(tool_dict['arguments']): {tool_dict['arguments']}, (type: {type(tool_dict['arguments'])})")
            assert isinstance(tool_dict["arguments"], dict), (
                "Tool call arguments must be a dictionary"
            )
        except Exception as e:
            print(ast.literal_eval(tool_call["arguments"]))
            print(f"{e.__class__.__name__}: {e}")
            print('tool_call', tool_call)
            print('message', message)
            raise e
        tool_content = f"{tool_call_bos}\n{json.dumps(tool_dict)}\n{tool_call_eos}"
        _new_msg_contents.append(tool_content)
        new_tools.append(tool_dict)
        old_texts.append(old_text)
        
    new_msg_content = "\n".join(_new_msg_contents)
    new_msg = {"role": role, "content": new_msg_content}
    return new_msg, new_tools, old_texts


def build_state_action_samples( 
    sample: dict[str, Any],
    sample_idx: int,
    all_tools_store: dict[str, Any],
    discount_factor: float = 0.9,
) -> list[dict[str, Any]]:
    """
    Build state-action samples from a single sample of the dataset
    """
    new_samples: list[dict[str, Any]] = []   # state, action, done, reward, return_, timestep, discount_factor
    running_chat: list[dict[str, Any]] = []  # (no next_obs)

    tools = [respond_user_tool] + sample["tools"]
    # Separate system prompt from messages
    _fewshot_str = "\n\n<example conversation>\n-\n</example conversation>\n\n"
    system_prompt = sample["messages"][0]["content"].replace(_fewshot_str, "").strip()

    messages = sample["messages"][1:]
    done = False
    timestep = 0
    # reward = 2 * int(sample["correct"]) - 1  # 1 if correct, -1 if incorrect
    reward = 1  # assume always correct

    # Get domain / policy
    domain = system_prompt.split("<policy>")[-1].strip().lower().split("agent")[0].replace("#", "").strip()

    for msg_idx, message in enumerate(messages):
        try:
            new_msg, new_tools, old_texts = convert_msg_to_think(message)
        except Exception as e:
            e_class = e.__class__.__name__
            print(f"Local error: Above {e_class}: {e}")
            print(json.dumps(message, indent=2))
            raise e
        
        if new_tools is not None and len(new_tools) > 0:
            for tool in new_tools:
                name = tool["name"]
                arguments = tool["arguments"]
                try:
                    argnames = list(arguments.keys())
                except Exception as e:
                    print(f"Error getting argnames: {e.__class__.__name__}: {e}")
                    print(tool)
                    raise e
                tool_hash = f"{name}_{'_'.join(argnames)}"
                if tool_hash not in all_tools_store:
                    all_tools_store[tool_hash] = {
                        "name": name,
                        "property_names": argnames,
                        "examples": [],
                        "count": 0
                    }
                # Add examples of tool usage
                if len(all_tools_store[tool_hash]["examples"]) < 3:
                    all_tools_store[tool_hash]["examples"].append(new_msg["content"])
                all_tools_store[tool_hash]["count"] += 1

        if new_msg is not None:
            running_chat.append(new_msg)

        # Store new samples
        # return_ = reward * (discount_factor ** (max_timestep - timestep))
        if msg_idx + 1 == len(messages):
            done = True

        if new_msg is not None and new_msg["role"] == "assistant":
            state = running_chat[:-1]
            action = running_chat[-1]
            new_sample = {
                "state": state,
                "action": action,
                "done": done,
                "reward": reward,
                "return_": None,       # to fill in
                "advantage": None,     # to fill in
                "timestep": timestep,
                "max_timestep": None,  # to fill in
                "discount_factor": discount_factor,
                "domain": domain,
                "system_prompt": system_prompt,
                "tools": tools,
                "unique_data_sample_id": sample_idx,
                "generation_id": 0,
            }
            # _old_kv = {k: v for k, v in sample.items() if k not in new_sample.keys()}
            # new_sample.update(_old_kv)
            new_samples.append(new_sample)
            timestep += 1

    # Update the return and max timesteps
    max_timestep = timestep - 1
    for _idx in range(len(new_samples)):
        return_ = reward * (discount_factor ** (max_timestep - new_samples[_idx]["timestep"]))
        new_samples[_idx]["return_"] = return_
        new_samples[_idx]["advantage"] = return_
        new_samples[_idx]["max_timestep"] = max_timestep
    
    return new_samples



def main():
    """
    Main function to build state-action samples from the dataset
    """
    # Load dataset from HF hub
    dataset_config = {
        "path": "amityco/apigen-tau-bench-split-turn",
        "cache_dir": "/scr/mzhang/data/",
        "split": "train"
    }
    ds = load_dataset(**dataset_config)

    all_samples = []
    chat_samples = []
    last_chat_len = 0
    # Go through and get only full trajectories
    for sample_idx, sample in enumerate(tqdm(ds, desc="Getting full trajectories", colour="green")):
        if len(sample["messages"]) < last_chat_len:  # indicates a new trajectory
            # max_timestep = len(chat_samples)
            # for _idx in chat_samples:
                # all_samples[_idx]["max_timestep"] = max_timestep
            all_samples.append(chat_samples[-1])
            chat_samples = []
        last_chat_len = len(sample["messages"])
        chat_samples.append(sample)

    ds = Dataset.from_list(all_samples)

    ALL_TOOLS = {}
    all_samples = []

    for sample_idx, sample in enumerate(ds):
        new_samples = build_state_action_samples(sample, sample_idx, ALL_TOOLS)
        all_samples.extend(new_samples)

    ds = Dataset.from_list(all_samples)
    

    # Split into "airline", "retail" splits and push to HuggingFace hub
    df = ds.to_pandas()
    ds_dict = {}
    for domain in df["domain"].unique():
        _df = df[df["domain"] == domain].reset_index(drop=True)
        ds_dict[domain] = Dataset.from_pandas(_df)
    ds_dict = DatasetDict(ds_dict)

    # hf_repo_path = "amityco/apigen-tau-bench-split-turn".lower()
    hf_repo_path = dataset_config["path"].lower()
    hf_repo_path = hf_repo_path.replace("/", "_").replace("-", "_")
    hf_repo_path = f"mzio/aprm-{hf_repo_path}"

    ds_dict.push_to_hub(hf_repo_path)
    hf_repo_url = f"https://huggingface.co/datasets/{hf_repo_path}" 
    print(f"-> Pushed to HuggingFace hub: {hf_repo_url}")
    return ds_dict


if __name__ == "__main__":
    main()
