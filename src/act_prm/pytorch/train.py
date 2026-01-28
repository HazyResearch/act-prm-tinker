"""
Training and evaluation functions
"""

from typing import Any, Callable

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from ..environments import Environment
from ..llm_handlers.huggingface import HuggingFaceLLM
from ..replay_buffer.types import Trajectory, TrajectoryGroup

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
) -> tuple[dict[str, Any], dict[str, list[Trajectory]]]:
    """
    Run rollouts for a single batch, e.g., by generating rollouts and grading them

    Returns:
    - final_metrics: Metrics for the batch, keyed by "{split}/{try_idx}/{metric}"
    - new_trajectories: Trajectories for the batch, keyed by an identifier (default "policy")
    """
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
    num_return_sequences = cfg.group_size if split == "train" else cfg.eval_group_size

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
        pbar_desc = f"Generating rollouts: sample {start_idx} / {start_idx + batch_size - 1}"
        sample_pbar = tqdm(
            range(start_idx, start_idx + batch_size),
            desc=pbar_desc,
            colour="blue",
            leave=False,
            position=pbar_position,
        )
        for _, sample_id in enumerate(sample_pbar):
            all_trajectory_groups.append(
                generator.do_group_rollout(
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
            pbar_desc = f"Generating rollouts: sample {start_idx} / {start_idx + batch_size - 1}"
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

    return final_metrics, new_trajectories
