"""
Training and evaluation functions
"""

# from copy import deepcopy
from typing import Any, Callable

from omegaconf import DictConfig
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizerBase

from ..environments import Environment
from ..llm_handlers.huggingface import HuggingFaceLLM
from ..replay_buffer.types import Trajectory, TrajectoryGroup

from .data import DataCollatorForPolicyGradient
from .generator import HuggingFaceGenerator


def run_rollouts(
    llm: HuggingFaceLLM,
    hf_tokenizer: PreTrainedTokenizerBase,
    generator_constructor: Callable[..., HuggingFaceGenerator],
    env: Environment,
    cfg: DictConfig,
    batch_id: int,
    checkpoint_name: str | None = None,
    split: str = "train",
    num_tries: int = 1,
    start_idx: int = 0,
    tasks_per_update: int | None = None,  # i.e., batch_size
    name_or_identifier: str | None = None,
    # Overrides for generation
    max_tokens: int | None = None,
    temperature: float | None = None,
    pbar_position: int = 0,
    num_return_sequences: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[Trajectory]]]:
    """
    Run rollouts for a single batch, e.g., by generating rollouts and grading them

    Returns:
    - final_metrics: Metrics for the batch, keyed by "{split}/{try_idx}/{metric}"
    - new_trajectories: Trajectories for the batch, keyed by an identifier (default "policy")
    """
    was_training = llm.model.training
    with torch.no_grad():
        llm.model.eval()
        env.split = split  # Select task split
        
        generator = generator_constructor(
            llm=llm,
            hf_tokenizer=hf_tokenizer,
            env=env,
            cfg=cfg,
            enable_thinking=cfg.get("enable_thinking", False),
            name_or_identifier=name_or_identifier,
        )
        batch_size = tasks_per_update or len(env)  # len(env) is the number of tasks or problems
        num_return_sequences = num_return_sequences or (
            cfg.group_size if split == "train" else cfg.eval_group_size
        )
        all_eval_metrics = {}
        keys_for_correct = []
        eval_metric_keys = [
            "final_reward", "first_return", "action_prob", "last_state_len",
            "timesteps", "correct", "total",
        ]
        # Store new trajectories to return
        new_trajectories: dict[str, list[Trajectory]] = {}

        assert num_tries == 1, "For this project, we only support one try 😊"
        for try_idx in range(num_tries):
            all_trajectory_groups: list[dict[str, list[TrajectoryGroup]]] = []
            pbar_desc = (
                f"Generating {num_return_sequences} rollouts for "
                f"sample {start_idx + 1} / {start_idx + batch_size}"
            )
            sample_pbar = tqdm(
                range(start_idx, start_idx + batch_size),
                desc=pbar_desc,
                colour="blue",
                leave=True,
                position=pbar_position,
            )
            for _, sample_id in enumerate(sample_pbar):
                all_trajectory_groups.append(
                    generator.do_group_rollout(
                        env=env,
                        sample_id=sample_id,
                        batch_id=batch_id,
                        split=split,
                        try_step=try_idx,
                        num_return_sequences=num_return_sequences,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        pbar_position=pbar_position + 1,
                    )
                )
                pbar_desc = (
                    f"Generating {num_return_sequences} rollouts for "
                    f"sample {start_idx + 1} / {start_idx + batch_size}"
                )
                sample_pbar.set_description(pbar_desc)

        # Save metrics and samples
        trajectory_keys = all_trajectory_groups[0].keys()
        _metric_prefix = f"{checkpoint_name}_{split}" if checkpoint_name is not None else split
        
        for _key in trajectory_keys:
            for trajectory_groups in all_trajectory_groups:     # list of list of trajectory groups
                for traj_group in trajectory_groups[_key]:      # len(trajectory_groups) usually 1,
                    for trajectory in traj_group.trajectories:  # can be >1, e.g., if step-wise adv
                        if _key == "policy":
                            # Only store metrics for the default "policy" trajectory group
                            for metric_key in eval_metric_keys:
                                _metric_key = f"{_metric_prefix}/try_{try_idx}/{metric_key}"
                                if metric_key == "correct":
                                    keys_for_correct.append(_metric_key)
                                if _metric_key not in all_eval_metrics:
                                    all_eval_metrics[_metric_key] = []
                                val = getattr(trajectory, metric_key, 1)  # 1 for total samples
                                all_eval_metrics[_metric_key].append(val)
                        # Add trajectory to list of new trajectories
                        if _key not in new_trajectories:
                            new_trajectories[_key] = []
                        new_trajectories[_key].append(trajectory)

    final_metrics = {}  # return these metrics for the batch
    # 1. Compute aggregate metrics
    for k, v in all_eval_metrics.items():
        if "correct" in k or "total" in k:
            final_metrics[k] = np.sum(v).item()  # convert to float for json.dumps
        else:
            final_metrics[k] = np.mean(v).item()
        final_metrics[f"{k}_std"] = np.std(v).item()
        final_metrics[f"{k}_max"] = np.max(v).item()
    # 2. Add accuracy (dummy for Act-PRM training rollouts)
    for k in keys_for_correct:
        total_v = final_metrics[k.replace("correct", "total")]
        final_metrics[k.replace("correct", "accuracy")] = final_metrics[k] / total_v

    if was_training:
        llm.model.train()

    return final_metrics, new_trajectories


def prepare_minibatch(
    new_trajectories: list[Trajectory],
    hf_tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 2,
    batch_idx: int = 0,  # for debugging
    **dataloader_kwargs: Any,
) -> tuple[DataLoader, dict[str, Any]]:
    """
    Convert a minibatch of trajectories to a PyTorch Dataloader
    """
    metrics = {}
    # Assemble training data
    data_dict: list[dict[str, list[float | int]]] = []
    for trajectory in new_trajectories:
        for episode_step in trajectory.episode_steps:
            sa_input_ids = episode_step.state_action_tokens
            act_logprobs = episode_step.old_logprobs  # 1. "full" action len, first token counts (maybe not)
            # input_tokens = sa_input_ids[:-1]
            # target_tokens = sa_input_ids[1:]
            state_len = episode_step.state_len
            target_state_len = episode_step.state_len - 1
            # 2. ^So target_state_len + len(act_logprobs) == len(sa_input_ids),  (maybe not)
            #    a bit different from Tinker where they don't predict first action token.
            padded_logprobs = [0.0] * target_state_len + act_logprobs
            adv = episode_step.advantage
            padded_advantages = [0.0] * target_state_len + [adv] * len(act_logprobs)
            padded_mask = [0] * target_state_len + [1] * len(act_logprobs)
            # Add labels to double-check
            sa_labels = [-100] * state_len + sa_input_ids[state_len:]
            # sa_labels = [-100] * target_state_len + sa_input_ids[target_state_len:]

            # print(f"batch_idx: {batch_idx}")
            # print("hf_tokenizer.decode(sa_input_ids[-len(act_logprobs):])")
            # print(hf_tokenizer.decode(sa_input_ids[-len(act_logprobs):]))
            # print("-" * 100)
            # print("hf_tokenizer.decode(sa_input_ids[target_state_len:])")
            # print(hf_tokenizer.decode(sa_input_ids[target_state_len:]))
            # print("-" * 100)
            try:
                assert (
                    len(sa_input_ids) - 1  # 3. not len(sa_input_ids) - 1
                    == len(padded_logprobs)
                    == len(padded_advantages)
                    == len(padded_mask)
                    == target_state_len + len(act_logprobs)
                )
            except AssertionError:
                print("len(sa_input_ids) is 1 more than expected".upper())
                print(f"len(sa_input_ids) - 1: {len(sa_input_ids) - 1}")
                print(f"len(padded_logprobs): {len(padded_logprobs)}")
                print(f"len(padded_advantages): {len(padded_advantages)}")
                print(f"len(padded_mask): {len(padded_mask)}")
                print(f"target_state_len: {target_state_len}")
                print(f"len(act_logprobs): {len(act_logprobs)}")
                breakpoint()

            data_dict.append({
                "input_ids": sa_input_ids,
                "attention_mask": [True] * len(sa_input_ids),
                "advantages": padded_advantages,  # Note that advantages and logprobs are already 
                "logprobs": padded_logprobs,      # shifted to account for next-token prediction
                "label_mask": padded_mask,
                "state_len": target_state_len,
                "action_len": len(act_logprobs),
                "labels": sa_labels,
            })

    dataset = HFDataset.from_list(data_dict)
    collate_fn = DataCollatorForPolicyGradient(tokenizer=hf_tokenizer, return_tensors="pt")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, **dataloader_kwargs
    )
    return dataloader, metrics  # empty metrics for now


def hide_observations(
    messages: list[dict[str, str]],
    hidden_obs_content: str = "...",
    first_obs_to_show: int = 2,  # e.g., to keep prompt
    last_obs_to_show: int = 1,   # e.g., to keep last observation
) -> list[dict[str, str]]:
    """
    Maybe hide past observations from messages
    """
    user_indices = [
        idx for idx, message in enumerate(messages) if message["role"] in ["user", "tool"]
    ]
    last_message_idx = user_indices[-last_obs_to_show] if last_obs_to_show > 0 else len(messages)
    return [
        {"role": message["role"], "content": hidden_obs_content}
        if (
            message["role"] in ["user", "tool"]
            and (idx >= first_obs_to_show and idx < last_message_idx)
        )
        else message
        for idx, message in enumerate(messages)
    ]
