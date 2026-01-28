"""
BabyAI-Text environment for ACT-PRM
"""

from __future__ import annotations

import json
import re
import logging
from copy import copy, deepcopy
from typing import Any, Annotated

from pydantic import ConfigDict, Field, SkipValidation

# Compat shim for gym 0.26.x on NumPy >= 2.0 (np.bool8 removed)
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

try:
    import gym
except ImportError as exc:
    raise ImportError(
        "gym is required for BabyAI-Text. "
        "See https://github.com/flowersteam/Grounding_LLMs_with_online_RL/tree/main/babyai-text"
    ) from exc

try:
    import babyai_text  # noqa: F401  # registers BabyAI-* envs with gym
    from babyai.bot import Bot
except ImportError as exc:
    raise ImportError(
        "babyai_text and babyai are required for BabyAI-Text. "
        "Install from https://github.com/flowersteam/Grounding_LLMs_with_online_RL/tree/main/babyai-text"
    ) from exc

from ..base import Environment
from ..types import EnvironmentState, EnvironmentStepResult
from ...llm_handlers import ActionFromLLM

from .prompts import get_instruction_prompt
from .tools import BabyAiTextTool

logger = logging.getLogger(__name__)

DEFAULT_ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]

MESSAGE_TEMPLATE = """## Observation:
{observation}

## Status:
{state_json}"""


class BabyAiTextState(EnvironmentState):
    """
    Pydantic state for BabyAI-Text episodes
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    sample_env: Annotated["gym.Env", SkipValidation] = Field(exclude=True, repr=False)
    tools: list[dict[str, Any]]
    action_space: list[str]
    action_trajectory: list[dict[str, str]]
    last_action_text: str
    mission: str
    last_description: str
    inventory: str | None


class BabyAiTextStepResult(EnvironmentStepResult):
    state: BabyAiTextState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


class BabyAiTextEnv(Environment):
    """
    BabyAI-Text environment implemented as our base Environment subclass
    """

    def __init__(
        self,
        env_name: str = "BabyAI-MixedTestLocal-v0",
        action_space: list[str] | None = None,
        num_train_samples: int = 200,
        num_val_samples: int = 32,
        num_test_samples: int = 32,
        max_turns: int = 20,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = "You are a helpful assistant that can answer questions and call tools.",
        **kwargs: Any,
    ) -> None:
        super().__init__(max_turns=max_turns, seed=seed, split=split, **kwargs)
        self.env_name = env_name
        self.action_space = action_space or DEFAULT_ACTION_SPACE
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.datasets = self.init_data()

        self.tool_helper = BabyAiTextTool()
        self.tool_descriptions = self.tool_helper.get_tool_descs()

    def init_data(self) -> dict[str, list[int]]:
        total_samples = self.num_train_samples + self.num_val_samples + self.num_test_samples
        all_seeds = list(range(total_samples))
        return {
            "train": all_seeds[:self.num_train_samples],
            "eval": all_seeds[self.num_train_samples:self.num_train_samples + self.num_val_samples],
            "test": all_seeds[self.num_train_samples + self.num_val_samples:],
        }

    def _build_system_prompt(self) -> str:
        instruction = get_instruction_prompt(max_turns=self.max_turns)
        return f"{self.system_prompt}\n\n{instruction}".strip()

    def _join_descriptions(self, descriptions: Any) -> str:
        if descriptions is None:
            return ""
        if isinstance(descriptions, list):
            return ". ".join([d for d in descriptions if d])
        return str(descriptions)

    def _get_inventory(self, env: gym.Env) -> str | None:
        carrying = getattr(env, "carrying", None)
        if carrying is None:
            return None
        return f"{carrying.color} {carrying.type}"

    def _build_player_prompt(self, mission: str, descriptions: Any) -> str:
        actions = ", ".join(self.action_space)
        desc_text = self._join_descriptions(descriptions)
        prompt = (
            "You are playing 'BabyAI-Text'.\n"
            f"Your goal is to {mission}.\n"
            f"Available actions are {actions}.\n"
            f"{desc_text}\n"
            "On your turn, call one of the available tools to take an action.\n"
        )
        if self.max_turns:
            prompt += f"The game lasts for {self.max_turns} turns in total.\n"
        return prompt

    def _format_state_content(
        self,
        observation: str,
        mission: str,
        inventory: str | None,
        last_action: str,
        timestep: int,
        reward: float | None = None,
    ) -> str:
        state_json = {
            "mission": mission,
            "inventory": inventory or "empty",
            "last_action": last_action,
            "timestep": timestep,
            "steps_left": max(self.max_turns - timestep, 0),
        }
        if reward is not None:
            state_json["reward"] = reward
        return MESSAGE_TEMPLATE.format(
            observation=observation.strip(),
            state_json=json.dumps(state_json, indent=2),
        )

    def _get_gold_path(self, env: gym.Env) -> list[dict[str, Any]]:
        """
        Get action/observation pairs from the BabyAI bot
        """
        steps: list[dict[str, Any]] = []
        try:
            env_copy = deepcopy(env)
            bot = Bot(env_copy)
            done = False
            action = None
            while not done:
                action = bot.replan(action)
                if str(action) == "Actions.done":
                    break
                action_idx = int(action)
                action_text = self.action_space[action_idx]
                _, reward, done, info = env_copy.step(action_idx)
                observation = self._join_descriptions(info.get("descriptions"))
                steps.append({
                    "action": action_text,
                    "observation": observation,
                    "inventory": self._get_inventory(env_copy),
                    "reward": reward,
                    "done": done,
                })
        except Exception as exc:
            logger.warning("Failed to build gold path: %s", exc)
        return steps

    def _build_action_trajectory(
        self,
        initial_prompt: str,
        mission: str,
        gold_steps: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "user", "content": initial_prompt}]
        for idx, step in enumerate(gold_steps, start=1):
            action_text = step["action"]
            tool_call = self.tool_helper.get_llm_toolcall_from_action(action_text)
            messages.append({"role": "assistant", "content": tool_call})
            tool_content = self._format_state_content(
                observation=step["observation"],
                mission=mission,
                inventory=step.get("inventory"),
                last_action=action_text,
                timestep=idx,
                reward=step.get("reward"),
            )
            messages.append({"role": "tool", "content": tool_content})
        return messages

    def reset(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
        try_step: int = 0,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> BabyAiTextState:
        sample_idx_adj = self.adjust_sample_idx(sample_idx)
        sample_seed = self.datasets[self.split][sample_idx_adj] + self.seed
        env = gym.make(self.env_name, seed=sample_seed)
        try:
            env.seed(sample_seed)
        except Exception:
            pass
        reset_result = env.reset()
        text_state: dict[str, Any] = {}
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            _, maybe_text_state = reset_result
            if isinstance(maybe_text_state, dict):
                text_state = maybe_text_state
        elif isinstance(reset_result, dict):
            text_state = reset_result

        mission = text_state.get("mission") or getattr(env, "mission", "")
        descriptions = text_state.get("descriptions") or text_state.get("description") or []
        initial_prompt = self._build_player_prompt(mission, descriptions)
        last_description = self._join_descriptions(descriptions)
        inventory = self._get_inventory(env)

        gold_steps = self._get_gold_path(env)
        action_trajectory = self._build_action_trajectory(
            initial_prompt=initial_prompt,
            mission=mission,
            gold_steps=gold_steps,
        )
        messages = [{"role": "user", "content": initial_prompt}]
        return BabyAiTextState(
            system_prompt=self._build_system_prompt(),
            new_messages=messages,
            model_response=None,
            prior_messages=[],
            tools=[],  # disable tokenizer tool injection; we format tool calls explicitly
            action_space=self.action_space,
            action_trajectory=action_trajectory,
            sample_env=env,
            last_action_text="",
            mission=mission,
            last_description=last_description,
            inventory=inventory,
            sample_id=sample_idx,
            generation_id=generation_idx,
            batch_id=batch_idx,
            try_step=try_step,
            timestep=0,
            metadata={"correct": 0, "total": 1},
            first_obs_to_show=len(messages) + 1,
        )

    def step(self, **kwargs: Any) -> BabyAiTextStepResult:
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: BabyAiTextState,
        current_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> BabyAiTextStepResult:
        done = False
        truncated = False
        reward = 0.0

        env = current_state.sample_env
        metadata = copy(current_state.metadata)
        timestep = int(current_state.timestep)
        try_step = int(current_state.try_step)
        updated_try_step = False

        env_messages: list[dict[str, Any]] = []
        stdout = ""
        tool_call_error = False
        last_action_text = ""
        last_description = current_state.last_description
        made_tool_call = False

        for action_idx, action in enumerate(parsed_actions):
            if action.type == "function_call":
                tool_name = action.name or ""
                action_text = self.tool_helper(tool_name=tool_name)
                call_id = action.call_id
            elif action.type in ["message", "reasoning"]:
                if action_idx + 1 == len(parsed_actions):
                    env_messages.append({
                        "role": "user",
                        "content": "Sad! You must persist and call tools to complete the task.",
                    })
                    done = True
                    truncated = True
                    if not updated_try_step:
                        try_step += 1
                        updated_try_step = True
                continue
            if action.type == "function_call":
                if action_text not in self.action_space:
                    stdout = (
                        "Invalid tool call. "
                        f"Tool '{tool_name}' is not a valid BabyAI action."
                    )
                    tool_call_error = True
                else:
                    action_id = self.action_space.index(action_text)
                    _, reward, done, info = env.step(action_id)
                    observation = self._join_descriptions(info.get("descriptions"))
                    last_action_text = action_text
                    last_description = observation
                    stdout = self._format_state_content(
                        observation=observation,
                        mission=current_state.mission,
                        inventory=self._get_inventory(env),
                        last_action=action_text,
                        timestep=timestep + 1,
                        reward=reward,
                    )
                    made_tool_call = True
                    if reward > 0:
                        metadata["correct"] = 1

                env_messages.append({
                    "role": "tool",
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": stdout,
                })

                if tool_call_error:
                    done = True
                    truncated = True
                    if not updated_try_step:
                        try_step += 1
                        updated_try_step = True
            else:
                logger.error("Invalid parsed action: %s", action)

        timestep = timestep + 1
        if timestep >= self.max_turns + 1:
            truncated = True
            done = True
            env_messages.append({"role": "user", "content": self.truncation_message})
            if not updated_try_step:
                try_step += 1
                updated_try_step = True

        if len(env_messages) == 0:
            env_messages.append({
                "role": "user",
                "content": "No tool calls were parsed. Please try again.",
            })
            logger.error("No tool calls were parsed.")

        current_messages = self.maybe_hide_observations(
            current_messages or [],
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )

        metadata.update({
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "made_tool_call": made_tool_call,
        })
        new_state = BabyAiTextState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages,
            tools=current_state.tools,
            action_space=current_state.action_space,
            action_trajectory=current_state.action_trajectory,
            sample_env=env,
            last_action_text=last_action_text,
            mission=current_state.mission,
            last_description=last_description,
            inventory=self._get_inventory(env),
            sample_id=current_state.sample_id,
            generation_id=current_state.generation_id,
            batch_id=current_state.batch_id,
            try_step=try_step,
            timestep=timestep,
            metadata=metadata,
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )
        return BabyAiTextStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=new_state.metadata,
        )


class AsyncBabyAiTextEnv(BabyAiTextEnv):
    """
    Async wrapper for BabyAI-Text environment
    """

    async def reset_async(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
        try_step: int = 0,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> BabyAiTextState:
        return super().reset(
            sample_idx=sample_idx,
            generation_idx=generation_idx,
            try_step=try_step,
            batch_idx=batch_idx,
            **kwargs,
        )

    async def step_async(self, **kwargs: Any) -> BabyAiTextStepResult:
        return super().step(**kwargs)
