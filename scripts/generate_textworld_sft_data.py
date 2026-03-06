"""
Generate SFT demonstration data from TextWorld ground-truth walkthroughs.

For each game in a TextWorld env config, steps through the ground-truth action
trajectory and records (state, action, tools) at each timestep. Saves the result
as a HuggingFace dataset that can be loaded by ActionLmEnv for SFT training.

Example:
```bash
uv run python scripts/generate_textworld_sft_data.py \
    --env_config textworld/coin_collector_medium \
    --output_dir ./data/textworld/sft/coin_collector_medium \
    --hide_observations \
    --actions_only
```
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from rich import print as rich_print
from tqdm import tqdm

from act_prm.environments.textworld.env import TextWorldEnv

logger = logging.getLogger(__name__)


def collect_demonstrations(
    env: TextWorldEnv,
    split: str = "train",
    actions_only: bool = False,
    hide_observations: bool = False,
    hidden_obs_content: str = "...",
    first_obs_to_show: int = 2,
    last_obs_to_show: int = 1,
) -> list[dict[str, Any]]:
    """
    Collect ground-truth demonstration data from a TextWorld environment.

    For each game, steps through the walkthrough and records state-action pairs
    at each timestep.

    Returns a list of sample dicts with keys:
        - state: list[dict] — chat messages up to (but not including) the action
        - action: dict — the assistant message (tool call)
        - tools: list[dict] — tool descriptions
        - unique_data_sample_id: int
        - timestep: int
        - generation_id: int (always 0)
        - return_: float (1.0 for successful walkthroughs)
        - advantage: float (1.0)
    """
    env.split = split
    sample_indices = env.datasets[split]
    samples: list[dict[str, Any]] = []

    pbar = tqdm(
        enumerate(sample_indices),
        total=len(sample_indices),
        desc=f"Collecting {split} demonstrations",
        colour="cyan",
    )
    for sample_idx, _ in pbar:
        state = env.reset(sample_idx=sample_idx)
        tools = state.tools
        walkthrough = state.action_trajectory  # list of tool call strings
        tw_walkthrough = state.tw_extra_walkthrough  # raw textworld actions

        # Build the full message sequence by stepping through the env
        # Start with the initial observation
        messages: list[dict[str, Any]] = list(state.new_messages)  # initial user msg

        for timestep, (action_text, tw_action) in enumerate(
            zip(walkthrough, tw_walkthrough)
        ):
            # The state is messages so far (what the model sees before generating)
            state_messages = list(messages)

            # Maybe hide past observations
            if hide_observations and len(state_messages) > first_obs_to_show:
                state_messages = _hide_observations(
                    state_messages,
                    hidden_obs_content=hidden_obs_content,
                    first_obs_to_show=first_obs_to_show,
                    last_obs_to_show=last_obs_to_show,
                )

            # The action is the assistant's tool call message
            action_msg = {"role": "assistant", "content": action_text}

            samples.append(
                {
                    "state": state_messages,
                    "action": action_msg,
                    "tools": tools,
                    "unique_data_sample_id": sample_idx,
                    "timestep": timestep,
                    "generation_id": 0,
                    "return_": 1.0,
                    "advantage": 1.0,
                }
            )

            # Now step the env to get the next observation
            # Step the underlying textworld env directly
            tw_game_state, tw_reward, tw_done = state.sample_tw_env.step(tw_action)

            # Build the next observation message
            state_json = {
                env._key(k): v
                for k, v in tw_game_state.items()
                if k in env.state_keys
            }
            state_json["max_possible_score"] = tw_game_state["max_score"]
            state_json["moves_left"] = env.max_turns - state_json["moves"]
            state_json["last_action"] = tw_action
            from act_prm.environments.textworld.env import MESSAGE_TEMPLATE

            obs_content = MESSAGE_TEMPLATE.format(
                action_feedback=tw_game_state["feedback"].strip(),
                state_json=json.dumps(state_json, indent=2),
            )

            # Add assistant action + env response to messages for next timestep
            messages.append(action_msg)
            messages.append(
                {
                    "role": "tool",
                    "type": "function_call_output",
                    "call_id": f"call_{timestep}",
                    "output": obs_content,
                }
            )

    return samples


def _hide_observations(
    messages: list[dict[str, str]],
    hidden_obs_content: str = "...",
    first_obs_to_show: int = 2,
    last_obs_to_show: int = 1,
) -> list[dict[str, str]]:
    """Hide past observations from messages."""
    user_indices = [
        idx
        for idx, message in enumerate(messages)
        if message["role"] in ["user", "tool"]
    ]
    last_message_idx = (
        user_indices[-last_obs_to_show] if last_obs_to_show > 0 else len(messages)
    )
    return [
        (
            {"role": message["role"], "content": hidden_obs_content}
            if (
                message["role"] in ["user", "tool"]
                and idx >= first_obs_to_show
                and idx < last_message_idx
            )
            else message
        )
        for idx, message in enumerate(messages)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TextWorld SFT data")
    parser.add_argument(
        "--env_config",
        type=str,
        required=True,
        help="Environment config path (e.g., textworld/coin_collector_medium)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the HuggingFace dataset",
    )
    parser.add_argument(
        "--actions_only",
        action="store_true",
        help="Strip reasoning traces, keep only tool calls",
    )
    parser.add_argument(
        "--hide_observations",
        action="store_true",
        help="Hide past observations (replace with '...')",
    )
    parser.add_argument(
        "--hidden_obs_content",
        type=str,
        default="...",
    )
    parser.add_argument("--first_obs_to_show", type=int, default=2)
    parser.add_argument("--last_obs_to_show", type=int, default=1)
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="HuggingFace Hub dataset name to push to (e.g., mzio/tw-sft-coin-medium)",
    )
    args = parser.parse_args()

    from omegaconf import OmegaConf

    env_cfg = OmegaConf.load(f"./configs/environments/{args.env_config}.yaml")
    env = TextWorldEnv(**env_cfg)

    # Collect demonstrations for train and test splits
    all_samples = []
    for split in ["train", "test"]:
        samples = collect_demonstrations(
            env,
            split=split,
            actions_only=args.actions_only,
            hide_observations=args.hide_observations,
            hidden_obs_content=args.hidden_obs_content,
            first_obs_to_show=args.first_obs_to_show,
            last_obs_to_show=args.last_obs_to_show,
        )
        for s in samples:
            s["split"] = split
        all_samples.extend(samples)
        rich_print(
            f"[bright_green]Collected {len(samples)} samples for {split} split[/bright_green]"
        )

    # Save as HuggingFace dataset
    ds = Dataset.from_list(all_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    rich_print(
        f"[bold bright_blue]Saved {len(ds)} samples to {output_dir}[/bold bright_blue]"
    )

    if args.push_to_hub:
        ds.push_to_hub(args.push_to_hub)
        rich_print(
            f"[bold bright_blue]Pushed to {args.push_to_hub}[/bold bright_blue]"
        )


if __name__ == "__main__":
    main()
