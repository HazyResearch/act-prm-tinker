"""
PyTorch Generator for Act-PRM with proper parallel rollouts (non-bandit setting)

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
    Act-PRM generator with proper parallel rollouts (non-bandit).

    Each of the N rollouts independently generates thoughts at every step,
    producing N full trajectories. The per-step reward is the probability
    of the target action given the generated thought: P(action | thought, state).
    This enables computing proper discounted returns across the full episode.

    Unlike the bandit version which picks the best thought at each step, all N
    rollouts run in parallel (batch generation + batch logprob computation),
    with done rollouts removed from the active batch as they complete.
    """

    def __init__(
        self,
        discount_factor: float = 1.0,
        **kwargs: Any,
    ) -> None:
        # Force reward_method to action_probs for nobandit
        kwargs.setdefault("reward_method", "action_probs")
        super().__init__(discount_factor=discount_factor, **kwargs)

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
        Generate N independent full rollouts for a single task sample, in parallel.

        All active rollouts are batch-generated and batch-scored at each step.
        Rollouts that finish early are removed from the active batch. This produces
        N complete trajectories with per-step rewards, enabling proper discounted
        return computation.

        Returns dict with keys:
        - "policy": TrajectoryGroups for (state, action, thought) -- RL training
        - "think_act_policy": TrajectoryGroups for (state, thought, action) -- SFT training
        """
        llm = llm or self.llm
        env = env or self.env
        cfg = cfg or self.cfg
        env.split = split
        discount_factor = (
            discount_factor
            if discount_factor is not None
            else (self.discount_factor or cfg.discount_factor)
        )

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
            # Per-rollout accumulators (indexed by generation id)
            all_policy_steps: list[list[EpisodeStep]] = [
                [] for _ in range(num_return_sequences)
            ]
            all_think_act_steps: list[list[EpisodeStep]] = [
                [] for _ in range(num_return_sequences)
            ]
            all_final_rewards: list[float] = [0.0] * num_return_sequences

            # Active rollout tracking -- shrinks as rollouts finish
            gen_ids_todo = list(range(num_return_sequences))
            batch_states: list[ActionProcessRewardState] = [
                env.reset(
                    sample_idx=sample_id,
                    generation_idx=gen_idx,
                    try_step=try_step,
                )
                for gen_idx in gen_ids_todo
            ]

            while len(gen_ids_todo) > 0:
                num_active = len(gen_ids_todo)

                # -- 1. Build thought prompts for all active rollouts --
                batch_state_messages: list[list[dict[str, Any]]] = [
                    self._get_messages_from_state(state) for state in batch_states
                ]
                batch_act_prompt_msgs: list[list[dict[str, Any]]] = []
                batch_input_ids_list: list[torch.Tensor] = []
                for state_msgs in batch_state_messages:
                    act_prompt_msgs, input_ids = self._get_thought_prompt(state_msgs)
                    batch_act_prompt_msgs.append(act_prompt_msgs)
                    batch_input_ids_list.append(input_ids.squeeze(0))

                # Left-pad to uniform length for batch generation
                max_input_len = max(ids.shape[0] for ids in batch_input_ids_list)
                padded_input_ids = torch.full(
                    (num_active, max_input_len),
                    hf_tokenizer.pad_token_id,
                    dtype=batch_input_ids_list[0].dtype,
                    device=device,
                )
                for i, ids in enumerate(batch_input_ids_list):
                    padded_input_ids[i, max_input_len - len(ids) :] = ids.to(device)

                # -- 2. Batch-generate thoughts --
                attention_mask = (padded_input_ids != hf_tokenizer.pad_token_id).long()
                outputs = llm.model.generate(
                    input_ids=padded_input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=hf_tokenizer.pad_token_id,
                )
                decoded_texts = hf_tokenizer.batch_decode(
                    outputs[:, max_input_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                thoughts = [self._parse_thoughts(txt) for txt in decoded_texts]

                # -- 3. Build standard chats and reward computation inputs --
                batch_standard_chats: list[list[dict[str, str]]] = []
                batch_state_thought_msgs: list[list[dict[str, str]]] = []
                batch_thought_act_msgs: list[list[dict[str, str]]] = []
                batch_state_thought_act_msgs: list[list[dict[str, str]]] = []

                for i, state in enumerate(batch_states):
                    first_msg_to_show = getattr(state, "first_obs_to_show", 0) - 3
                    standard_system_prompt = (
                        state.original_system_prompt or env.original_system_prompt
                    )
                    standard_chat = process_state_messages_for_metrics(
                        batch_state_messages[i],
                        system_prompt=standard_system_prompt,
                        first_msg_to_show=max(first_msg_to_show, 0),
                    )
                    batch_standard_chats.append(standard_chat)

                    batch_state_thought_msgs.append(
                        standard_chat + [{"role": "assistant", "content": thoughts[i]}]
                    )
                    thought_act_msg = {
                        "role": "assistant",
                        "content": f"{thoughts[i]}\n\n{state.action_target}",
                    }
                    batch_thought_act_msgs.append([thought_act_msg])
                    batch_state_thought_act_msgs.append(
                        standard_chat + [thought_act_msg]
                    )

                # -- 4. Batch-compute action probability rewards --
                _reward_outputs = get_act_logprobs_and_state_act_tokens_from_messages(
                    model=llm.model,
                    hf_tokenizer=hf_tokenizer,
                    state_messages=batch_state_thought_msgs,
                    state_action_messages=batch_state_thought_act_msgs,
                    state_continue_final_message=True,
                )
                act_logprobs: list[list[float]] = _reward_outputs[0]
                # Per-step reward: mean action logprob exponentiated.
                # We skip _compute_group_rewards since each rollout is independent;
                # group-level normalization happens via TrajectoryGroup.
                rewards = [np.exp(sum(lps) / max(len(lps), 1)) for lps in act_logprobs]

                # -- 5. Batch-compute SFT-style (state, thought, action) logprobs --
                _sft_outputs = get_act_logprobs_and_state_act_tokens_from_messages(
                    model=llm.model,
                    hf_tokenizer=hf_tokenizer,
                    state_messages=batch_standard_chats,
                    state_action_messages=batch_state_thought_act_msgs,
                    state_continue_final_message=False,
                )
                sft_logprobs: list[list[float]] = _sft_outputs[0]
                sft_tokens: list[list[int]] = _sft_outputs[1]

                sft_state_lens = [
                    len(
                        hf_tokenizer.apply_chat_template(
                            sc,
                            add_generation_prompt=True,
                            continue_final_message=False,
                            tokenize=True,
                        )
                    )
                    for sc in batch_standard_chats
                ]

                # -- 6. Batch-compute action-prompted (state, action, thought) logprobs --
                batch_sat_msgs: list[list[dict[str, Any]]] = []
                for i in range(num_active):
                    sat_msgs = deepcopy(batch_act_prompt_msgs[i])
                    sat_msgs[-1]["content"] += decoded_texts[i]
                    batch_sat_msgs.append(sat_msgs)

                _policy_outputs = get_act_logprobs_and_state_act_tokens_from_messages(
                    model=llm.model,
                    hf_tokenizer=hf_tokenizer,
                    state_messages=batch_act_prompt_msgs,
                    state_action_messages=batch_sat_msgs,
                    state_continue_final_message=True,
                )
                policy_logprobs: list[list[float]] = _policy_outputs[0]
                policy_tokens: list[list[int]] = _policy_outputs[1]

                policy_state_lens = [ids.shape[0] for ids in batch_input_ids_list]

                # -- 7. Step environments and build episode steps --
                done_indices: list[int] = []
                for i in range(num_active):
                    gen_id = gen_ids_todo[i]
                    state = batch_states[i]
                    reward = rewards[i]

                    model_messages = [{"role": "assistant", "content": thoughts[i]}]
                    parsed_actions = get_actions(model_messages)
                    env_step_result: EnvironmentStepResult = env.step(
                        parsed_actions=parsed_actions,
                        current_state=state,
                        current_messages=batch_state_messages[i],
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
                        "generation_id": gen_id,
                        "split": split,
                        "action_prob": reward,
                    }

                    # (state, thought, action) episode step -- for SFT
                    think_act_step = EpisodeStep(
                        state=batch_standard_chats[i],
                        action=batch_thought_act_msgs[i][0],
                        state_action_tokens=sft_tokens[i],
                        state_len=sft_state_lens[i],
                        old_logprobs=sft_logprobs[i],
                        **shared_kwargs,
                    )
                    all_think_act_steps[gen_id].append(think_act_step)

                    # (state, action, thought) episode step -- for RL
                    policy_step = EpisodeStep(
                        state=batch_state_messages[i],
                        action={"role": "assistant", "content": decoded_texts[i]},
                        state_action_tokens=policy_tokens[i],
                        state_len=policy_state_lens[i],
                        old_logprobs=policy_logprobs[i],
                        **shared_kwargs,
                    )
                    all_policy_steps[gen_id].append(policy_step)

                    all_final_rewards[gen_id] = reward

                    # Verbose display
                    if self.verbose and gen_id < self.samples_to_display:
                        try:
                            max_turns = len(state.assistant_indices)
                        except AttributeError:
                            max_turns = len(state.action_trajectory)
                        header_text = (
                            f"Batch {batch_id}, Split {split}, Try {try_step}, "
                            f"Sample {sample_id}, Gen {gen_id}, "
                            f"Step {state.timestep} / {max_turns - 1}, "
                            f"Reward {reward:.4f}"
                        )
                        panel_parts = [
                            f"Reward: [bright_green]{reward:.4f}[/bright_green]",
                            f"Run url: [cyan]{self.run_url}[/cyan]",
                            f"Run cmd: [bright_blue]{self.run_cmd}[/bright_blue]",
                        ]
                        if self.name_or_identifier:
                            panel_parts.append(
                                f"Name/ID: [bright_yellow]{self.name_or_identifier}[/bright_yellow]"
                            )
                        panel_content = "\n".join(panel_parts)
                        try:
                            sft_text = hf_tokenizer.decode(
                                sft_tokens[i],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=True,
                            )
                            rich_print_messages(
                                msg_text=sft_text,
                                bos_token="<|im_start|>",
                                eos_token="<|im_end|>\n",
                                assistant_color=ROYGBIV[gen_id % len(ROYGBIV)],
                            )
                            console.print(
                                Panel(panel_content, title=header_text, style="bold")
                            )
                        except Exception as e:
                            logger.error(f"{e.__class__.__name__}: {e}")
                            print(f"{'-' * 50} {header_text} {'-' * 50}")
                            print(panel_content)

                    # Update state or mark done
                    if done:
                        done_indices.append(i)
                    else:
                        batch_states[i] = next_state

                # Remove finished rollouts in reverse order to preserve indices
                for idx in reversed(done_indices):
                    gen_ids_todo.pop(idx)
                    batch_states.pop(idx)

            # -- Build multi-step Trajectory objects with proper returns --
            policy_trajectories = [
                Trajectory(
                    episode_steps=all_policy_steps[gen_id],
                    try_step=try_step,
                    discount_factor=discount_factor,
                    final_reward=all_final_rewards[gen_id],
                )
                for gen_id in range(num_return_sequences)
            ]
            think_act_trajectories = [
                Trajectory(
                    episode_steps=all_think_act_steps[gen_id],
                    try_step=try_step,
                    discount_factor=discount_factor,
                    final_reward=all_final_rewards[gen_id],
                )
                for gen_id in range(num_return_sequences)
            ]

            policy_group = self._get_trajectory_group(
                trajectories=policy_trajectories,
                final_rewards=all_final_rewards,
                discount_factor=discount_factor,
            )
            think_act_group = self._get_trajectory_group(
                trajectories=think_act_trajectories,
                final_rewards=all_final_rewards,
                discount_factor=discount_factor,
            )

        if was_training:
            llm.model.train()

        hf_tokenizer.padding_side = og_tokenizer_padding_side
        return {
            "policy": [policy_group],
            "think_act_policy": [think_act_group],
        }
