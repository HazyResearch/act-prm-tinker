"""
Tinker Generator with Action Process Reward Models
"""

import asyncio
import logging
from copy import deepcopy
from typing import Any

from tinker.types import ModelInput
from transformers import PreTrainedTokenizerBase

from ..environments.base import Environment
from ..environments.act_prm import ActionProcessRewardState
from ..environments.types import EnvironmentStepResult
from ..llm_handlers.action_utils import get_actions
from ..llm_handlers.tinker import TinkerCompleter, TokensWithLogprobsAndText
from ..llm_handlers.types import ActionFromLLM
from ..replay_buffer.types import EpisodeStep, Trajectory, TrajectoryGroup

from .tinker_act_prm import (
    compute_group_thought_action_metrics,
    process_state_messages_for_metrics,
)
from .tinker_act_prompt_aprm import (
    TinkerActionPromptActPrmGenerator,
)
logger = logging.getLogger(__name__)


class TinkerActionPromptNoBanditActPrmGenerator(TinkerActionPromptActPrmGenerator):
    """
    Tinker Generator with Action Process Reward Models and proper returns (non-bandit setting)
    """
    def __init__(self, discount_factor: float = 1.0, **kwargs: Any) -> None:
        super().__init__(discount_factor=discount_factor, **kwargs)
        self.reward_method = "action_probs"

    async def do_single_rollout(
        self,
        llm: TinkerCompleter | None = None,
        env: Environment | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        split: str = "train",
        batch_id: int = 0,
        unique_data_sample_id: int = 0,
        generation_id: int = 0,
        try_step: int = 0,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Trajectory]:
        """
        Generate a full rollout in the environment, and return the trajectory
        """
        llm = llm or self.llm
        env = env or self.env
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer

        max_tokens = max_tokens or llm.max_tokens
        temperature = temperature or llm.temperature

        episode_steps: list[EpisodeStep] = []
        action_prompt_episode_steps: list[EpisodeStep] = []
        state: ActionProcessRewardState = await env.reset_async(
            sample_idx=unique_data_sample_id,
            generation_idx=generation_id,
            try_step=try_step,
        )
        max_turns = len(state.assistant_indices)

        done = False
        reward = 0.0
        while not done:
            # Generate model responses and step through the environment
            state_messages: list[dict[str, Any]] = self._get_messages_from_state(state)
            # Prompt for thoughts
            act_prompt_state_messages, input_ids = self._get_thought_prompt(state_messages)
            tinker_input = ModelInput.from_ints(input_ids)
            # 1. Generate model responses (thoughts + actions)
            response: TokensWithLogprobsAndText = await llm.generate(
                tinker_input,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            # Parse thoughts from responses
            # -> demonstrations may have <thought>...</thought>
            # Singletons so we can reuse functions in ./tinker_act_prompt_aprm.py
            thoughts_in_group: list[str] = [self._parse_thoughts(response.text)]
            # Compute per-step rewards for each thought
            # # -> Get current state without last action
            # standard_chat = process_state_messages_for_metrics(state_messages, state.system_prompt)
            # -> Get current state without last action, remove few-shot prompts
            first_msg_to_show = getattr(state, "first_obs_to_show", 0) - 3
            # ^-1 ActPRM environment previously counts system prompt as first message,
            # but we apply after system_prompt in process_state_messages_for_metrics
            standard_chat = process_state_messages_for_metrics(
                state_messages, state.system_prompt,
                first_msg_to_show=max(first_msg_to_show, 0)
            )
            group_metrics = await compute_group_thought_action_metrics(
                state_messages=standard_chat,
                generated_thoughts=thoughts_in_group,
                target_action=state.action_target,
                tools=state.tools,
                hf_tokenizer=hf_tokenizer,
                sampling_client=llm.sampling_client,
            )

            # Process generations and take next environment step
            model_messages = [{"role": "assistant", "content": thoughts_in_group[0]}]
            parsed_actions: list[ActionFromLLM] = get_actions(model_messages)
            try:
                _action = parsed_actions[0]
            except IndexError as e:
                logger.error(f"IndexError: {e}")
                breakpoint()
  
            env_step_result: EnvironmentStepResult = await env.step_async(
                parsed_actions=parsed_actions,
                # model_response=model_messages,
                current_state=state,
                current_messages=state_messages,
            )
            next_state = env_step_result.state
            truncated  = env_step_result.truncated
            done       = env_step_result.done
            next_obs = [
                {
                    "role": msg["role"],
                    "content": msg["output"] if msg.get("output", None) else msg["content"]
                } for msg in next_state.new_messages
            ]

            # ---------- Save episode steps for each generation ----------
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
                "unique_data_sample_id": unique_data_sample_id,
                "generation_id": generation_id,
                "split": split,
            }

            # 1. Get state-thought-action artifacts (for SFT; thought_action_policy)
            reward += self._compute_group_rewards(
                rewards_in_group=group_metrics["target_action_probs"],
                split=split,
            )[0]
            thought_action_messages = group_metrics["thought_action_messages"][0]
            state_action_tokens = group_metrics["state_thought_action_tokens"][0]
            thought_action_logps = group_metrics["action_logprobs"][0]
            thought_action_state_len = group_metrics["state_len"][0]

            episode_step = EpisodeStep(
                state=standard_chat,  # messages without last action
                action=thought_action_messages[0],  # dict[str, str]
                state_action_tokens=state_action_tokens,
                state_len=thought_action_state_len,
                old_logprobs=thought_action_logps,
                **shared_kwargs,
            )
            episode_steps.append(episode_step)
        
            # 2. Get state-action-thought artifacts (for RL; (action-thought) policy
            state_action_thought_messages = deepcopy(act_prompt_state_messages)
            state_action_thought_messages[-1]["content"] += response.text
            state_action_thought_tokens = hf_tokenizer.apply_chat_template(
                state_action_thought_messages,
                # continue_final_message=True,  # don't add eos_token to final message
                add_generation_prompt=False,
                tokenize=True,
                # tools=state.tools, # no tools for action-prompted generation
            )
            action_token_len = len(state_action_thought_tokens) - len(input_ids)
            state_action_thought_tinker_input = ModelInput.from_ints(state_action_thought_tokens)
            act_thought_logps = await llm.compute_logprobs_async(state_action_thought_tinker_input)
            act_thought_logps = act_thought_logps[-action_token_len:]
            action_msgs = [{"role": "assistant", "content": response.text}]

            action_prompt_episode_step = EpisodeStep(
                state=state_messages,  # state,
                action=action_msgs[0],  # dict[str, str]
                state_action_tokens=state_action_thought_tokens,
                state_len=len(input_ids),
                old_logprobs=act_thought_logps,
                **shared_kwargs,
            )
            action_prompt_episode_steps.append(action_prompt_episode_step)

            # If verbose, rich print the state and action
            if self.verbose and generation_id < self.samples_to_display:
                _header_text = (
                    f"Batch {batch_id}, Split {split}, Try {try_step}, "
                    f"Sample {unique_data_sample_id}, Generation {generation_id}, "
                    f"Step {state.timestep} (Max {max_turns - 1})"
                )
                panel_content = "\n".join([
                    f"Rewards: [bright_green][{reward:.4f}][/bright_green]",
                    f"Run url: [cyan]{self.run_url}[/cyan]",
                    f"Run cmd: [bright_blue]{self.cmd_str}[/bright_blue]",
                ])
                self.display_state_action_next_obs(  # slightly coded for Qwen models for now
                    state_messages=standard_chat,
                    action_messages=thought_action_messages,
                    next_obs_messages=[],
                    hf_tokenizer=hf_tokenizer,
                    tools=state.tools,
                    header_text=f"SFT trajectory:\n{_header_text}",
                    panel_content=panel_content,
                    generation_id=generation_id,
                )
                self.display_state_action_next_obs(  # slightly coded for Qwen models for now
                    state_messages=act_prompt_state_messages,
                    action_messages=action_msgs,
                    next_obs_messages=[],
                    hf_tokenizer=hf_tokenizer,
                    tools=state.tools,
                    header_text=f"Action-prompt Completion:\n{_header_text}",
                    panel_content=panel_content,
                    generation_id=generation_id,
                )
            # Transition to next state
            state = env_step_result.state

        thought_action_trajectory = Trajectory(
            episode_steps=episode_steps,
            try_step=try_step,
            discount_factor=self.discount_factor,
            final_reward=reward,
        )
        action_prompt_trajectory = Trajectory(
            episode_steps=action_prompt_episode_steps,
            try_step=try_step,
            discount_factor=self.discount_factor,
            final_reward=reward,
        )
        return {
            "policy": action_prompt_trajectory,
            "thought_action_policy": thought_action_trajectory,
        }

    async def do_group_rollout(
        self,
        num_return_sequences: int,
        **single_rollout_kwargs: Any,
    ) -> dict[str, list[TrajectoryGroup]]:
        """
        Generate a group of trajectories in the environment, and return a list of the trajectory
        group(s).

        By default, we should just return a singleton with 1 TrajectoryGroup.
        """
        trajectories_in_group: list[dict[str, Trajectory]] = await asyncio.gather(
            *[
                self.do_single_rollout(generation_id=gen_idx, **single_rollout_kwargs)
                for gen_idx in range(num_return_sequences)
            ],
        )
        # (state, action, thought) samples for policy RL
        all_trajectory_groups: list[TrajectoryGroup] = [
            self._get_trajectory_group(
                trajectories=[traj["policy"] for traj in trajectories_in_group], 
                discount_factor=self.discount_factor,
            )
        ]
        # (state, thought, action) samples for SFT
        all_thought_action_trajectory_groups: list[TrajectoryGroup] = [
            self._get_trajectory_group(
                trajectories=[traj["thought_action_policy"] for traj in trajectories_in_group], 
                discount_factor=self.discount_factor,
            )
        ]
        return {
            "policy": all_trajectory_groups,
            "thought_action_policy": all_thought_action_trajectory_groups, 
        }
