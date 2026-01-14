"""
Tinker Generator with Action Process Reward Models
"""

import asyncio
from typing import Any

from ..replay_buffer.types import Trajectory, TrajectoryGroup
from ..trainer.utils import gather_with_progress

from .tinker import TinkerGenerator


def process_state_messages_for_act_prm(
    state_messages: list[dict[str, str]],
    system_prompt: dict[str, str],
) -> list[dict[str, str]]:
    """
    Ensure state messages have 1 system prompt and no last assistant message
    """
    if state_messages[0]["role"] == "system":
        state_messages = state_messages[1:]
    if state_messages[-1]["role"] == "assistant":
        state_messages = state_messages[:-1]
    return [system_prompt] + state_messages


async def compute_single_thought_action_metrics(
    state_messages: list[dict[str, str]],
    generated_thought: str,
    target_action: str,
    hf_tokenizer: PreTrainedTokenizerBase,
    sampling_client: SamplingClient,
    tools: list[dict[str, str]] | None = None,
) -> dict[str, float | list[int] | list[float] | int]:
    """
    Compute logprob and likelihood metrics for a given state, thought, and target action

    Returns a dictionary with the following keys:
    - action_probs:                  p(action | state, generated_thought)
    - action_logprobs:               logprobs of the action tokens
    - state_thought_action_tokens:   tokens of the entire (state, thought, action) trajectory
    - state_thought_action_logprobs: logprobs of the entire (state, thought, action) trajectory
    - state_thought_len:             number of (state, thought) tokens
    - action_len:                    number of action tokens
    """
    _tokenize_kwargs = {
        "add_generation_prompt": False,
        "tokenize": True,
        "tools": tools,
    }
    generated_thought = generated_thought.split("</thought>")[0].strip()
    thought_msgs = [{"role": "assistant", "content": generated_thought}]
    thought_action_msgs = [{
        "role": "assistant",
        "content": f"{generated_thought}\n\n{target_action}",
    }]
    prefix_tokens = hf_tokenizer.apply_chat_template(
        state_messages + thought_msgs,
        continue_final_message=True,
        **_tokenize_kwargs,
    )
    full_tokens = hf_tokenizer.apply_chat_template(
        state_messages + thought_action_msgs,
        continue_final_message=False,
        **_tokenize_kwargs,
    )
    action_token_len = len(full_tokens) - len(prefix_tokens)

    # Compute length-normalized joint probabilities of action tokens as reward metrics
    tinker_prompt = ModelInput.from_ints(full_tokens)
    logprobs = await sampling_client.compute_logprobs_async(tinker_prompt)
    action_logprobs = np.array(logprobs[-action_token_len:])
    action_probs = np.exp(action_logprobs.sum() / len(action_logprobs)).item()  # length-normalize

    return {
        "action_probs": action_probs,
        "action_logprobs": action_logprobs.tolist(),
        "state_thought_action_tokens": full_tokens,
        "state_thought_action_logprobs": logprobs,
        "state_thought_len": len(prefix_tokens),
        "action_len": action_token_len,
    }


def compute_group_thought_action_metrics(
    state_messages: list[dict[str, str]],
    generated_thoughts: list[str],
    target_action: str,
    system_prompt: dict[str, str],
    tools: list[dict[str, str]],
    hf_tokenizer: PreTrainedTokenizerBase,
    sampling_client: SamplingClient,
) -> dict[str, list[Any]]:
    """
    Compute thought-action metrics for a group of generated thoughts

    Returns a dictionary with the following keys:
    - action_probs:                  p(action | state, generated_thought)
    - action_logprobs:               logprobs of the action tokens
    - state_thought_action_tokens:   tokens of the entire (state, thought, action) trajectory
    - state_thought_action_logprobs: logprobs of the entire (state, thought, action) trajectory
    - state_thought_len:             number of (state, thought) tokens
    - action_len:                    number of action tokens
    - state_len:                     number of state tokens
    """
    state_messages = process_state_messages_for_act_prm(state_messages, system_prompt)
    state_len = len(
        hf_tokenizer.apply_chat_template(
            state_messages,
            add_generation_prompt=True,
            tokenize=True,
            tools=tools,
        )
    )
    metrics_in_group: list[dict[str, Any]] = await gather_with_progress(
        [
            compute_single_thought_action_metrics(
                state_messages=state_messages,
                generated_thought=gen_thought,
                target_action=target_action,
                hf_tokenizer=hf_tokenizer,
                sampling_client=sampling_client,
                tools=tools,
            ) for gen_thought in generated_thoughts
        ],
        desc="Computing thought-action metrics, p(action | state, thought)",
        colour="green",
    )
    # Convert list of dicts to dict of lists
    metrics_by_key: dict[str, Any] = {}
    for k, v in metrics_in_group[0].items():
        metrics_by_key[k] = [getattr(m, k) for m in metrics_in_group]
    metrics_by_key["state_len"] = [state_len] * len(generated_thoughts)
    return metrics_by_key


class TinkerActPrmGenerator(TinkerGenerator):
    """
    Tinker Generator with Action Process Reward Models
    """
    async def do_single_rollout(self, **kwargs: Any) -> Trajectory:
        """
        Not implemented for Act-PRM Generators
        """
        raise NotImplementedError("do_single_rollout not implemented for Act-PRM Generators")

    async def do_group_rollout(
        self,
        num_return_sequences: int,
        llm: TinkerCompleter | None = None,
        env: Environment | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        split: str = "train",
        batch_id: int = 0,
        unique_data_sample_id: int = 0,
        try_step: int = 0,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[TrajectoryGroup]:
        """
        Generate thought-action trajectories given observed actions in an Act-PRM environment.

        Unlike typical group-rollouts, at *each* step, we:
        1. Generate `num_return_sequences` thoughts,
        2. Compute the per-step reward for each generation,
        3. Pick the highest-reward thought to continue for the next step.

        This results in *one* full trajectory (from start to workflow completion).

        However for training, we still save each (state, action', thought', reward') tuple 
        for all `num_return_sequences` thoughts as a TrajectoryGroup. This results in returning
        `num_steps` TrajectoryGroups.
        """
        llm = llm or self.llm
        env = env or self.env
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer

        max_tokens = max_tokens or llm.max_tokens
        temperature = temperature or llm.temperature

        state = await env.reset_async(
            sample_idx=unique_data_sample_id,
            generation_idx=0,
            try_step=try_step,
        )
        done = False
        while not done:
            # Generate model responses and step through the environment
            state_messages: list[dict[str, Any]] = self._get_messages_from_state(state)
            # Prompt for thoughts, e.g., of the form <action_bos>(action)</action_eos><thought_bos>
            input_ids: list[int] = hf_tokenizer.apply_chat_template(
                state_messages,
                add_generation_prompt=False,
                continue_final_message=True,  # remove added eos_token
                tokenize=True,
            )
            tinker_input = ModelInput.from_ints(input_ids)
            # Generate `num_return_sequences` thoughts
            responses_in_group: list[TokensWithLogprobsAndText] = await asyncio.gather(*[
                llm.generate(
                    tinker_input,
                    max_tokens=max_tokens,
                    temperature=temperature
                ) for _ in range(num_return_sequences)
            ])
            # Parse thoughts from responses
            thoughts_in_group: list[str] = [response[0].text for response in responses_in_group]
            # Compute per-step rewards for each thought
            standard_chat = state.thought_action_chat[:-1]  # Get current state (no last action
            group_metrics = compute_group_thought_action_metrics(
                state_messages=standard_chat,
                generated_thoughts=thoughts_in_group,
                target_action=state.action_target,
                system_prompt=state.system_prompt,
                tools=state.tools,
                hf_tokenizer=hf_tokenizer,
                sampling_client=llm.sampling_client,
            )
            rewards_in_group = group_metrics["action_probs"]

            # Save episode steps for each generation
            episode_steps_in_group: list[EpisodeStep] = [
                EpisodeStep(
                    state=state_messages,
                    # Be sure to handle specifically
                    action={"role": "assistant", "content": thoughts_in_group[i]},
                    next_obs=[],
                    tools=state.tools,
                    state_len=len(input_ids),
                    # To handle
                    state_action_tokens=group_metrics["state_thought_action_tokens"][i],
                    old_logprobs=group_metrics["action_logprobs"][i],
                    temperature=temperature,
                    reward=rewards_in_group[i],
                    done=done,
                    truncated=truncated,
                    timestep=state.timestep,
                    try_step=try_step,
                    batch_id=batch_id,
                    unique_data_sample_id=unique_data_sample_id,
                    generation_id=i,
                    split=split,
                ) for i in range(num_return_sequences)
            ]
            trajectories_in_group: list[Trajectory] = [
                Trajectory(
                    episode_steps=[episode_steps_in_group[i]],
                    try_step=try_step,
                    discount_factor=self.discount_factor,
                    final_state=state_messages,
                    final_obs=[],
                    final_reward=rewards_in_group[i],
                )
                for i in range(num_return_sequences)
            ]
            trajectory_group: TrajectoryGroup = self._get_trajectory_group(
                trajectories=trajectories_in_group,
                final_rewards=rewards_in_group,
                discount_factor=self.discount_factor,
            )
            
            # Pick the highest-reward thought to continue for the next step
            best_thought_idx = np.argmax(rewards_in_group)
            best_thought = thoughts_in_group[best_thought_idx]

            env_step_result: EnvironmentStepResult = await env.step_async(
                parsed_actions=parsed_actions,
                model_response=model_messages,
                current_state=state,
                # Set next_state.prior_messages to all messages
                current_messages=state_messages,
            )
            next_state = env_step_result.state
            






        states_in_group: list[EnvironmentState] = await asyncio.gather(*[
            env.reset_async(
                sample_idx=unique_data_sample_id,
                generation_idx=gen_idx,
                try_step=try_step,
            ) for gen_idx in range(num_return_sequences)
        ])
        state_messages_in_group: list[list[dict[str, Any]]] = [
            self._get_messages_from_state(state) for state in states_in_group
        ]

        state: EnvironmentState = await env.reset_async(
            sample_idx=unique_data_sample_id,
            generation_idx=generation_id,
            try_step=try_step,
        )
        
        
        
        trajectories_in_group: list[Trajectory] = await asyncio.gather(
            *[
                self.do_single_rollout(generation_id=gen_idx, **single_rollout_kwargs)
                for gen_idx in range(num_return_sequences)
            ],
        )
        # final_rewards_in_group: list[float] = [t.final_reward for t in trajectories_in_group]
        return [
            self._get_trajectory_group(
                trajectories=trajectories_in_group, 
                # final_rewards=final_rewards_in_group,
                discount_factor=self.discount_factor,
            )
        ]
