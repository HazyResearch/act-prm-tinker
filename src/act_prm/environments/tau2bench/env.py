"""
Online tau2-bench environment for step-by-step rollouts.

Wraps tau2's AgentGymEnv to provide a live interactive environment where an
LLM agent interacts with a simulated user and domain-specific tools.
This enables rollout evaluation and RL training on tau2-bench tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from copy import copy
from typing import Any, Annotated

from pydantic import ConfigDict, Field, SkipValidation

from ..base import Environment
from ..types import EnvironmentState, EnvironmentStepResult
from ...llm_handlers import ActionFromLLM
from .utils import convert_tau2_tools, parse_observation

logger = logging.getLogger(__name__)


class Tau2BenchState(EnvironmentState):
    """
    State for a tau2-bench episode.

    Holds a reference to the live AgentGymEnv instance (SkipValidation because
    it's not a Pydantic-serializable object), along with task metadata and
    the latest info dict from tau2.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Live tau2 gym environment for this episode (excluded from serialization)
    tau2_gym_env: Annotated[Any, SkipValidation] = Field(exclude=True, repr=False)
    # Current task identifier
    task_id: str
    # Info dict from the last tau2 gym step (contains tools, policy, reward_info)
    tau2_info: Annotated[dict[str, Any], SkipValidation] = Field(
        default_factory=dict, exclude=True, repr=False
    )


class Tau2BenchStepResult(EnvironmentStepResult):
    """Result of a step in the tau2-bench environment."""

    state: Tau2BenchState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


class Tau2BenchEnv(Environment):
    """
    Online tau2-bench environment.

    Creates a fresh AgentGymEnv per episode, routes parsed LLM actions through
    tau2's orchestrator (which handles user simulation, tool execution, and
    evaluation), and maps results back to our Environment interface.

    Args:
        domain: tau2 domain name (e.g. "airline", "retail", "telecom").
        user_llm: Model name for the simulated user (e.g. "gpt-4.1-2025-04-14").
        user_llm_args: LLM arguments for user simulator (e.g. {"temperature": 0.0}).
        task_split: Optional tau2 task split name ("train", "test", or None for all).
        num_train_tasks: Number of tasks allocated to the train split.
        num_val_tasks: Number of tasks for the validation/eval split (None = use test).
        num_test_tasks: Number of tasks allocated to the test split.
        max_steps: Maximum steps per episode in tau2's orchestrator.
        max_turns: Maximum LLM turns before we truncate (our-level truncation).
        seed: Random seed.
        split: Active data split ("train", "eval", or "test").
        system_prompt: Base system prompt for the agent.
    """

    def __init__(
        self,
        domain: str = "airline",
        user_llm: str = "gpt-4.1-2025-04-14",
        user_llm_args: dict[str, Any] | None = None,
        task_split: str | None = None,
        num_train_tasks: int = 100,
        num_val_tasks: int | None = None,
        num_test_tasks: int = 100,
        max_steps: int = 50,
        # Inherited arguments
        max_turns: int = 30,
        num_tries: int = 1,
        eval_num_tries: int = 1,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = "You are a helpful customer service agent.",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            max_turns=max_turns,
            num_tries=num_tries,
            eval_num_tries=eval_num_tries,
            seed=seed,
            split=split,
            **kwargs,
        )
        self.domain = domain
        self.user_llm = user_llm
        self.user_llm_args = user_llm_args or {"temperature": 0.0}
        self.task_split = task_split
        self.num_train_tasks = num_train_tasks
        self.num_val_tasks = num_val_tasks
        self.num_test_tasks = num_test_tasks
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.eval_num_tries = eval_num_tries

        # Load tasks and split into train/eval/test
        self.datasets = self._init_data()

        # Extract tool descriptions and domain policy from a temporary AgentGymEnv
        self.tool_descriptions, self.domain_policy = self._init_tools_and_policy()

        # Build full system prompt incorporating domain policy
        self._full_system_prompt = self.build_system_prompt()

    def _init_data(self) -> dict[str, list[Any]]:
        """
        Load tau2 tasks and split them into train/eval/test.

        Uses tau2's `get_tasks` with `task_split_name` to load the canonical
        train/test splits from tau2's split_tasks.json. This ensures splits
        match tau2's official task assignments:
          - airline: 30 train, 20 test (out of 50 total)
          - retail:  74 train, 40 test (out of 114 total)

        If `task_split` is set (e.g. "train" or "test"), loads only that
        split. If None, loads train and test splits separately and combines.

        The eval split defaults to the test split unless `num_val_tasks` is
        set, in which case the last `num_val_tasks` train tasks are used.

        Returns:
            Dict mapping split name to list of tau2 Task objects.
        """
        from tau2.run import get_tasks

        if self.task_split is not None:
            # Legacy behavior: load a single split
            all_tasks = get_tasks(
                task_set_name=self.domain,
                task_split_name=self.task_split,
            )
            test_tasks = all_tasks[: self.num_test_tasks]
            remaining = all_tasks[self.num_test_tasks :]
            train_tasks = remaining[: self.num_train_tasks]
            eval_tasks = test_tasks
        else:
            # Load tau2's canonical train/test splits
            train_tasks = list(
                get_tasks(
                    task_set_name=self.domain,
                    task_split_name="train",
                )
            )
            test_tasks = list(
                get_tasks(
                    task_set_name=self.domain,
                    task_split_name="test",
                )
            )
            eval_tasks = test_tasks  # default: eval = test

        # Optionally carve out a validation set from the end of train
        if self.num_val_tasks is not None:
            eval_tasks = train_tasks[-self.num_val_tasks :]
            train_tasks = train_tasks[: -self.num_val_tasks]

        # Optionally limit train/test sizes
        if self.num_train_tasks < len(train_tasks):
            train_tasks = train_tasks[: self.num_train_tasks]
        if self.num_test_tasks < len(test_tasks):
            test_tasks = test_tasks[: self.num_test_tasks]

        datasets = {
            "train": train_tasks,
            "eval": eval_tasks,
            "test": test_tasks,
        }
        for split_name, split_tasks in datasets.items():
            logger.info(
                f"tau2bench [{self.domain}] {split_name}: {len(split_tasks)} tasks "
                f"(IDs: {[t.id for t in split_tasks[:5]]}{'...' if len(split_tasks) > 5 else ''})"
            )
        return datasets

    def _init_tools_and_policy(self) -> tuple[list[dict[str, Any]], str]:
        """
        Extract tool descriptions and domain policy from a temporary AgentGymEnv.

        Creates a throwaway environment for the first available task, resets it
        to get the info dict (which contains tools and policy), then closes it.

        Returns:
            Tuple of (flat tool descriptions list, domain policy string).
        """
        from tau2.gym.gym_agent import AgentGymEnv

        # Use the first test task (always available) to extract tools/policy
        sample_task = self.datasets["test"][0]
        tmp_env = AgentGymEnv(
            domain=self.domain,
            task_id=sample_task.id,
            user_llm=self.user_llm,
            user_llm_args=self.user_llm_args,
            max_steps=self.max_steps,
        )
        _, info = tmp_env.reset()
        tau2_tools = info.get("tools", [])
        policy = info.get("policy", "")
        tool_descriptions = convert_tau2_tools(tau2_tools)
        tmp_env.close()
        return tool_descriptions, policy

    def build_system_prompt(self, system_prompt: str | None = None) -> str:
        """
        Build the full system prompt incorporating domain policy and respond_user instruction.

        Combines the base system prompt with the domain policy and an instruction
        to always use the respond_user tool for user-facing messages.

        Like TextWorldEnv.build_system_prompt(), accepts an optional system_prompt
        override so that trace configs (act_prm, act_lm) can inject their own
        base prompt while still getting the domain-specific policy.

        Args:
            system_prompt: Optional override for the base system prompt.
                If None, uses self.system_prompt from the config.

        Returns:
            Complete system prompt string.
        """
        system_prompt = system_prompt or self.system_prompt
        parts = [system_prompt]

        if self.domain_policy:
            parts.append(f"\n\n<policy>\n{self.domain_policy}\n</policy>")

        # Instruct agent to use respond_user for all user-facing messages
        parts.append(
            "\n\nIMPORTANT: When you want to respond or send a message to the user, "
            "you MUST use the `respond_user` tool with your message as the `text` argument. "
            "Do NOT send plain text messages — always use tool calls."
        )
        return "".join(parts)

    def __len__(self) -> int:
        """Get the number of tasks for the current split."""
        return len(self.datasets[self.split])

    def reset(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
        try_step: int = 0,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> Tau2BenchState:
        """
        Reset environment by creating a fresh AgentGymEnv for the selected task.

        Args:
            sample_idx: Index into the current split's task list (wraps around).
            generation_idx: Generation index (for multi-generation rollouts).
            try_step: Retry step counter.
            batch_idx: Batch index.

        Returns:
            Initial Tau2BenchState with tools, system prompt, and first user message.
        """
        # Close any previous episode's environment to prevent resource leaks
        if hasattr(self, "_current_tau2_env") and self._current_tau2_env is not None:
            try:
                self._current_tau2_env.close()
            except Exception as e:
                logger.warning(f"tau2bench: failed to close previous env: {e}")
            self._current_tau2_env = None

        sample_idx_adj = sample_idx % len(self.datasets[self.split])
        task = self.datasets[self.split][sample_idx_adj]

        # Create fresh tau2 gym environment for this episode
        from tau2.gym.gym_agent import AgentGymEnv

        tau2_env = AgentGymEnv(
            domain=self.domain,
            task_id=task.id,
            user_llm=self.user_llm,
            user_llm_args=self.user_llm_args,
            max_steps=self.max_steps,
        )
        self._current_tau2_env = tau2_env  # track for cleanup on next reset
        obs_str, info = tau2_env.reset(seed=self.seed + sample_idx)

        # Parse observation to get initial user message
        user_content = parse_observation(obs_str) if obs_str else "Hello, I need help."
        messages = [{"role": "user", "content": user_content}]

        if self.verbose and sample_idx == 0 and generation_idx == 0 and try_step == 0:
            logger.info(
                f"tau2bench reset: task_id={task.id}, obs={user_content[:100]}..."
            )

        return Tau2BenchState(
            system_prompt=self._full_system_prompt,
            new_messages=messages,
            model_response=None,
            prior_messages=[],
            tools=self.tool_descriptions,
            # tau2-specific fields
            tau2_gym_env=tau2_env,
            task_id=task.id,
            tau2_info=info,
            # Step-wise metadata
            sample_id=sample_idx,
            generation_id=generation_idx,
            batch_id=batch_idx,
            try_step=try_step,
            timestep=0,
            metadata={"correct": 0, "total": 1},
            first_obs_to_show=len(messages)
            + 1,  # keep system + initial user message visible
        )

    def step(self, **kwargs: Any) -> Tau2BenchStepResult:
        """Perform one step through the tau2-bench environment."""
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: Tau2BenchState,
        current_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Tau2BenchStepResult:
        """
        Step through the environment by routing parsed actions to tau2's gym env.

        Iterates over parsed_actions (following TextWorld's per-action loop pattern):
        - For `respond_user` calls: extracts text, sends to tau2, parses user response
        - For regular tool calls: formats as JSON, sends to tau2, gets tool result
        - For plain text (fallback): sends directly to tau2

        Args:
            parsed_actions: List of ActionFromLLM objects from the LLM's response.
            model_response: Raw model response (stored in state for provenance).
            current_state: Current Tau2BenchState from the previous step.
            current_messages: Full message history (for observation hiding).

        Returns:
            Tau2BenchStepResult with new state, reward, done, and truncated flags.
        """
        done = False
        truncated = False
        reward = 0.0

        tau2_env = current_state.tau2_gym_env
        metadata = copy(current_state.metadata)
        timestep = int(current_state.timestep)
        try_step = int(current_state.try_step)
        updated_try_step = False
        tau2_info = current_state.tau2_info

        env_messages: list[dict[str, Any]] = []
        made_tool_call = False

        for action_idx, action in enumerate(parsed_actions):
            if action.type == "function_call":
                fc_name = action.name
                fc_args = action.arguments or {}

                if fc_name == "respond_user":
                    # Send user-facing message through tau2's orchestrator
                    user_text = fc_args.get("text", "")
                    obs, step_reward, terminated, step_truncated, info = tau2_env.step(
                        user_text
                    )
                    tau2_info = info

                    # Tool result confirming message was sent
                    env_messages.append(
                        {
                            "role": "tool",
                            "type": "function_call_output",
                            "call_id": action.call_id,
                            "output": "Message sent to user.",
                        }
                    )

                    reward = float(step_reward)
                    done = bool(terminated) or bool(step_truncated)
                    if step_truncated:
                        truncated = True

                    # If not done, add the user's response as a new user message
                    if not done:
                        user_response = parse_observation(obs)
                        if user_response:
                            env_messages.append(
                                {
                                    "role": "user",
                                    "content": user_response,
                                }
                            )
                    made_tool_call = True

                else:
                    # Regular tool call: format as JSON and send to tau2
                    action_str = json.dumps(
                        {
                            "name": fc_name,
                            "arguments": fc_args,
                        }
                    )
                    try:
                        obs, step_reward, terminated, step_truncated, info = (
                            tau2_env.step(action_str)
                        )
                        tau2_info = info

                        # Parse tool result from observation
                        tool_output = parse_observation(obs) if obs else "No output."
                        env_messages.append(
                            {
                                "role": "tool",
                                "type": "function_call_output",
                                "call_id": action.call_id,
                                "output": tool_output,
                            }
                        )

                        reward = float(step_reward)
                        done = bool(terminated) or bool(step_truncated)
                        if step_truncated:
                            truncated = True
                        made_tool_call = True

                    except Exception as e:
                        error_msg = (
                            f"Tool call error for '{fc_name}': {type(e).__name__}: {e}"
                        )
                        logger.warning(error_msg)
                        env_messages.append(
                            {
                                "role": "tool",
                                "type": "function_call_output",
                                "call_id": action.call_id,
                                "output": error_msg,
                            }
                        )
                        done = True
                        truncated = True
                        if not updated_try_step:
                            try_step += 1
                            updated_try_step = True

                # Stop processing further actions if the episode ended
                if done:
                    break

            elif action.type in ["message", "reasoning"]:
                # Plain text or reasoning — only act on the last one
                if action_idx + 1 == len(parsed_actions):
                    # Fallback: send plain text to tau2 as a user-facing message
                    text = action.text or ""
                    if text.strip():
                        obs, step_reward, terminated, step_truncated, info = (
                            tau2_env.step(text)
                        )
                        tau2_info = info
                        reward = float(step_reward)
                        done = bool(terminated) or bool(step_truncated)
                        if step_truncated:
                            truncated = True

                        if not done:
                            user_response = parse_observation(obs)
                            if user_response:
                                env_messages.append(
                                    {
                                        "role": "user",
                                        "content": user_response,
                                    }
                                )
                    else:
                        # Empty text — treat as failure
                        done = True
                        truncated = True
                        env_messages.append(
                            {
                                "role": "user",
                                "content": "You must use tool calls to interact. Please try again.",
                            }
                        )
                        if not updated_try_step:
                            try_step += 1
                            updated_try_step = True
            else:
                logger.error(
                    f"Unknown action type '{action.type}' at index {action_idx}: {action}"
                )

        # Update timestep, check turn limit
        timestep += 1
        if timestep >= self.max_turns + 1:
            truncated = True
            done = True
            env_messages.append({"role": "user", "content": self.truncation_message})
            if not updated_try_step:
                try_step += 1
                updated_try_step = True

        # Ensure we always have at least one response message
        if len(env_messages) == 0:
            env_messages.append(
                {
                    "role": "user",
                    "content": "No actions were parsed. Please use tool calls to continue.",
                }
            )
            logger.error("No actions parsed in tau2bench step.")

        # Handle observation hiding for prior messages
        current_messages = self.maybe_hide_observations(
            current_messages or [],
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )

        # Track correctness: reward >= 1.0 means task solved
        if reward >= 1.0:
            metadata["correct"] = 1

        metadata.update(
            {
                "reward": reward,
                "done": done,
                "truncated": truncated,
                "made_tool_call": made_tool_call,
                "task_id": current_state.task_id,
            }
        )

        new_state = Tau2BenchState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages,
            tools=current_state.tools,
            # tau2-specific fields
            tau2_gym_env=tau2_env,
            task_id=current_state.task_id,
            tau2_info=tau2_info,
            # Step-wise metadata
            sample_id=current_state.sample_id,
            generation_id=current_state.generation_id,
            batch_id=current_state.batch_id,
            try_step=try_step,
            timestep=timestep,
            metadata=metadata,
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )
        return Tau2BenchStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=new_state.metadata,
        )


class AsyncTau2BenchEnv(Tau2BenchEnv):
    """
    Asynchronous wrapper for the tau2-bench environment.

    Uses asyncio.to_thread for reset and step since tau2's AgentGymEnv
    involves I/O-bound LLM calls in its orchestrator thread.
    """

    async def reset_async(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
        try_step: int = 0,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> Tau2BenchState:
        """
        Asynchronous reset — delegates to sync reset via asyncio.to_thread
        since tau2's orchestrator involves LLM API calls.
        """
        return await asyncio.to_thread(
            super().reset,
            sample_idx=sample_idx,
            generation_idx=generation_idx,
            try_step=try_step,
            batch_idx=batch_idx,
            **kwargs,
        )

    async def step_async(self, **kwargs: Any) -> Tau2BenchStepResult:
        """
        Asynchronous step — delegates to sync step via asyncio.to_thread
        since tau2's orchestrator involves LLM API calls.
        """
        return await asyncio.to_thread(super().step, **kwargs)
