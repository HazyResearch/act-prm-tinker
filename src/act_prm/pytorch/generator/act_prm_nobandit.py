"""
PyTorch Generator for Act-PRM with proper sequential rollouts (non-bandit setting)

Instead of generating N thoughts at each step, scoring them, and picking the best
to commit to (bandit), we run N independent full rollouts in parallel.
The reward at each step is P(action | thought, state), so returns at step 0 are
the discounted sum of all per-step action probabilities.
"""

import logging
from copy import copy, deepcopy
from typing import Any

from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from act_prm.environments import Environment, EnvironmentStepResult
from act_prm.environments.act_prm import ActionProcessRewardState

from act_prm.generator.tinker_act_prm import process_state_messages_for_metrics

from act_prm.llm_handlers import HuggingFaceLLM
from act_prm.llm_handlers.action_utils import get_actions
from act_prm.llm_handlers.types import ActionFromLLM
from act_prm.replay_buffer.types import (
    EpisodeStep,
    Trajectory,
    TrajectoryGroup,
)
from act_prm.utils.display import rich_print_messages

from .act_prm import (
    ActionPromptActPrmGenerator,
    get_act_logprobs_and_state_act_tokens_from_messages,
)


console = Console()
logger = logging.getLogger(__name__)
ROYGBIV = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"]


class NoBanditActionPromptActPrmGenerator(ActionPromptActPrmGenerator):
    """
    Act-PRM generator with proper sequential rollouts (non-bandit).

    Each of the N rollouts independently generates thoughts at every step,
    producing N full trajectories. The per-step reward is the probability
    of the target action given the generated thought: P(action | thought, state).
    This enables computing proper discounted returns across the full episode.
    """

    def __init__(
        self,
        discount_factor: float = 1.0,
        **kwargs: Any,
    ) -> None:
        # Force reward_method to action_probs for nobandit
        kwargs.setdefault("reward_method", "action_probs")
        super().__init__(discount_factor=discount_factor, **kwargs)

    def _do_single_rollout(
        self,
        gen_idx: int,
        llm: HuggingFaceLLM,
        env: Environment,
        hf_tokenizer: PreTrainedTokenizerBase,
        cfg: DictConfig,
        sample_id: int,
        batch_id: int,
        split: str,
        try_step: int,
        max_tokens: int,
        temperature: float,
        device: torch.device,
    ) -> dict[str, Trajectory]:
        """
        Run a single full sequential rollout, returning both policy and think_act_policy
        trajectories with per-step action-probability rewards.
        """
        # Episode step accumulators for both trajectory types
        policy_episode_steps: list[EpisodeStep] = []  # (state, action, thought)
        think_act_episode_steps: list[EpisodeStep] = []  # (state, thought, action)

        state: ActionProcessRewardState = env.reset(
            sample_idx=sample_id,
            generation_idx=gen_idx,
            try_step=try_step,
        )
        try:
            max_turns = len(state.assistant_indices)
        except AttributeError:
            max_turns = len(state.action_trajectory)

        done = False
        reward = 0.0

        while not done:
            state_messages: list[dict[str, Any]] = self._get_messages_from_state(state)

            # Prompt for thoughts: append action + <thought> tag to prompt continuation
            act_prompt_state_messages, input_ids = self._get_thought_prompt(
                state_messages
            )
            input_ids = input_ids.to(device)

            # Generate thought completion
            outputs = llm.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=hf_tokenizer.pad_token_id,
            )
            state_input_len = input_ids.shape[1]
            decoded_text = hf_tokenizer.decode(
                outputs[0, state_input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            thought = self._parse_thoughts(decoded_text)

            # Build the standard chat (without few-shot, without last action)
            first_msg_to_show = getattr(state, "first_obs_to_show", 0) - 3
            standard_system_prompt = (
                state.original_system_prompt or env.original_system_prompt
            )
            standard_chat: list[dict[str, str]] = process_state_messages_for_metrics(
                state_messages,
                system_prompt=standard_system_prompt,
                first_msg_to_show=max(first_msg_to_show, 0),
            )

            # Compute action probability reward: P(action | thought, state)
            state_thought_messages = [
                standard_chat + [{"role": "assistant", "content": thought}]
            ]
            thought_act_messages: list[list[dict[str, str]]] = [
                [
                    {
                        "role": "assistant",
                        "content": f"{thought}\n\n{state.action_target}",
                    }
                ]
            ]
            state_thought_act_messages: list[list[dict[str, str]]] = [
                standard_chat + _msgs for _msgs in thought_act_messages
            ]
            _outputs = get_act_logprobs_and_state_act_tokens_from_messages(
                model=llm.model,
                hf_tokenizer=hf_tokenizer,
                state_messages=state_thought_messages,
                state_action_messages=state_thought_act_messages,
                state_continue_final_message=True,
            )
            act_logprobs: list[list[float]] = _outputs[0]
            # Per-step reward: mean action logprob exponentiated
            reward = np.exp(sum(act_logprobs[0]) / max(len(act_logprobs[0]), 1))

            # Step environment
            model_messages = [{"role": "assistant", "content": thought}]
            parsed_actions: list[ActionFromLLM] = get_actions(model_messages)
            env_step_result: EnvironmentStepResult = env.step(
                parsed_actions=parsed_actions,
                current_state=state,
                current_messages=state_messages,
            )
            next_state = env_step_result.state
            done = env_step_result.done
            truncated = env_step_result.truncated
            next_obs = [
                {
                    "role": msg["role"],
                    "content": msg["output"]
                    if msg.get("output", None)
                    else msg["content"],
                }
                for msg in next_state.new_messages
            ]

            shared_kwargs = {
                "next_obs": next_obs,
                "tools": state.tools,
                "temperature": temperature,
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "timestep": state.timestep,
                "try_step": try_step,
                "batch_id": batch_id,
                "unique_data_sample_id": sample_id,
                "generation_id": gen_idx,
                "split": split,
                "action_prob": reward,
            }

            # 1. (state, thought, action) episode step — for SFT training
            # Get state_len for standard_chat with generation prompt
            state_len = len(
                hf_tokenizer.apply_chat_template(
                    standard_chat,
                    add_generation_prompt=True,
                    continue_final_message=False,
                    tokenize=True,
                )
            )
            # Recompute logprobs with continue_final_message=False for SFT-style tokens
            _sft_outputs = get_act_logprobs_and_state_act_tokens_from_messages(
                model=llm.model,
                hf_tokenizer=hf_tokenizer,
                state_messages=[standard_chat],
                state_action_messages=state_thought_act_messages,
                state_continue_final_message=False,
            )
            think_act_logprobs: list[float] = _sft_outputs[0][0]
            state_thought_act_tokens: list[int] = _sft_outputs[1][0]

            think_act_step = EpisodeStep(
                state=standard_chat,
                action=thought_act_messages[0][0],
                state_action_tokens=state_thought_act_tokens,
                state_len=state_len,
                old_logprobs=think_act_logprobs,
                **shared_kwargs,
            )
            think_act_episode_steps.append(think_act_step)

            # 2. (state, action, thought) episode step — for RL training
            state_action_thought_messages = deepcopy(act_prompt_state_messages)
            state_action_thought_messages[-1]["content"] += decoded_text
            state_act_thought_tokens = hf_tokenizer.apply_chat_template(
                state_action_thought_messages,
                add_generation_prompt=False,
                continue_final_message=True,
                tokenize=True,
            )

            # Compute logprobs for the thought tokens in action-prompted form
            _act_prompt_outputs = get_act_logprobs_and_state_act_tokens_from_messages(
                model=llm.model,
                hf_tokenizer=hf_tokenizer,
                state_messages=[act_prompt_state_messages],
                state_action_messages=[state_action_thought_messages],
                state_continue_final_message=True,
            )
            act_thought_logprobs: list[float] = _act_prompt_outputs[0][0]

            policy_step = EpisodeStep(
                state=state_messages,
                action={"role": "assistant", "content": decoded_text},
                state_action_tokens=state_act_thought_tokens,
                state_len=state_input_len,
                old_logprobs=act_thought_logprobs,
                **shared_kwargs,
            )
            policy_episode_steps.append(policy_step)

            # Verbose display
            if self.verbose and gen_idx < self.samples_to_display:
                header_text = (
                    f"Batch {batch_id}, Split {split}, Try {try_step}, "
                    f"Sample {sample_id}, Generation {gen_idx}, "
                    f"Step {state.timestep} / {max_turns - 1}, "
                    f"Reward {reward:.4f}"
                )
                panel_content = [
                    f"Reward: [bright_green]{reward:.4f}[/bright_green]",
                    f"Run url: [cyan]{self.run_url}[/cyan]",
                    f"Run cmd: [bright_blue]{self.run_cmd}[/bright_blue]",
                ]
                if self.name_or_identifier:
                    panel_content.append(
                        f"Name/ID: [bright_yellow]{self.name_or_identifier}[/bright_yellow]"
                    )
                panel_content = "\n".join(panel_content)
                try:
                    state_thought_act_text = hf_tokenizer.decode(
                        state_thought_act_tokens,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )
                    rich_print_messages(
                        msg_text=state_thought_act_text,
                        bos_token="<|im_start|>",
                        eos_token="<|im_end|>\n",
                        assistant_color=ROYGBIV[gen_idx % len(ROYGBIV)],
                    )
                    console.print(Panel(panel_content, title=header_text, style="bold"))
                except Exception as e:
                    logger.error(f"{e.__class__.__name__}: {e}")
                    print(f"{'-' * 50} {header_text} {'-' * 50}")
                    print(panel_content)

            # Transition
            state = next_state

        # Build Trajectory objects with per-step rewards for proper return computation
        policy_trajectory = Trajectory(
            episode_steps=policy_episode_steps,
            try_step=try_step,
            discount_factor=self.discount_factor,
            final_reward=reward,
        )
        think_act_trajectory = Trajectory(
            episode_steps=think_act_episode_steps,
            try_step=try_step,
            discount_factor=self.discount_factor,
            final_reward=reward,
        )
        return {
            "policy": policy_trajectory,
            "think_act_policy": think_act_trajectory,
        }

    def do_group_rollout(
        self,
        sample_id: int,
        batch_id: int,
        llm: HuggingFaceLLM | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        env: Environment | None = None,
        cfg: DictConfig | None = None,
        split: str = "train",
        try_step: int = 0,
        discount_factor: float | None = None,
        num_return_sequences: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        pbar_position: int = 0,
    ) -> dict[str, list[TrajectoryGroup]]:
        """
        Generate N independent full rollouts for a single task sample.

        Unlike the bandit version which picks the best thought at each step,
        each rollout runs independently from start to finish. This produces
        N complete trajectories with per-step rewards, enabling proper
        discounted return computation.

        Returns dict with keys:
        - "policy": TrajectoryGroups for (state, action, thought) — RL training
        - "think_act_policy": TrajectoryGroups for (state, thought, action) — SFT training
        """
        llm = llm or self.llm
        env = env or self.env
        cfg = cfg or self.cfg
        env.split = split
        discount_factor = discount_factor or self.discount_factor or cfg.discount_factor

        was_training = llm.model.training
        llm.model.eval()
        device = llm.model.device

        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        og_tokenizer_padding_side = copy(hf_tokenizer.padding_side)
        hf_tokenizer.padding_side = "left"

        max_tokens = max_tokens or cfg.max_tokens
        temperature = temperature or cfg.temperature
        num_return_sequences = num_return_sequences or (
            cfg.group_size if split == "train" else cfg.eval_group_size
        )

        with torch.no_grad():
            # Run N independent rollouts sequentially
            all_trajectories: list[dict[str, Trajectory]] = []
            for gen_idx in range(num_return_sequences):
                traj_dict = self._do_single_rollout(
                    gen_idx=gen_idx,
                    llm=llm,
                    env=env,
                    hf_tokenizer=hf_tokenizer,
                    cfg=cfg,
                    sample_id=sample_id,
                    batch_id=batch_id,
                    split=split,
                    try_step=try_step,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    device=device,
                )
                all_trajectories.append(traj_dict)

            # Group trajectories
            policy_trajectories = [t["policy"] for t in all_trajectories]
            think_act_trajectories = [t["think_act_policy"] for t in all_trajectories]

            policy_group = self._get_trajectory_group(
                trajectories=policy_trajectories,
                final_rewards=[t.final_reward for t in policy_trajectories],
                discount_factor=discount_factor,
            )
            think_act_group = self._get_trajectory_group(
                trajectories=think_act_trajectories,
                final_rewards=[t.final_reward for t in think_act_trajectories],
                discount_factor=discount_factor,
            )

        if was_training:
            llm.model.train()

        hf_tokenizer.padding_side = og_tokenizer_padding_side
        return {
            "policy": [policy_group],
            "think_act_policy": [think_act_group],
        }
