# src/<your_pkg>/environments/textworld/env.py

from __future__ import annotations

import json
import logging
from copy import copy
from typing import Any, Annotated

from pydantic import ConfigDict, Field, SkipValidation
from rich import print as rich_print
from textworld.core import (
    GameState,
    Environment as TextWorldEnvironment,
    Wrapper as TextWorldWrapper,
)

from ..base import Environment
from ..types import EnvironmentState, EnvironmentStepResult
from ...llm_handlers import ActionFromLLM  # same as your HotpotQA import

from .factory import TextWorldFactory
from .prompts import get_instruction_prompt
from .prompts_aprm import FEWSHOT_PROMPTS as ACT_PRM_FEWSHOT_PROMPTS
from .tools import TextWorldTool

logger = logging.getLogger(__name__)


DEFAULT_MAX_TURNS_BY_TASK = {
    "coin_collector": 25,
    "the_cooking_game": 80,
    "treasure_hunter": 40,
}

# Use to format the next observation
MESSAGE_TEMPLATE = """{action_feedback}

## Current State:

{state_json}"""

# Shared TextWorld GameState attributes across tasks that we'll keep track of
SHARED_GAMESTATE_KEYS = {
    "feedback",
    "extra.walkthrough",
    "objective",
    "max_score",
    "description",
    "inventory",
    "score",
    "moves",
    "won",
    "lost",
}


class TextWorldState(EnvironmentState):
    """
    Pydantic state for TextWorld episodes
    """
    # Allow TextWorldEnvironment | TextWorldWrapper attributes
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Keep track of the actual TextWorld env instance for each sample task
    # -> See Factory.make_env() in ./factory.py
    sample_tw_env: Annotated[
        "TextWorldEnvironment | TextWorldWrapper",
        SkipValidation,
    ] = Field(exclude=True, repr=False)

    tools: list[dict[str, Any]]
    action_trajectory: list[str]  # list of TextWorld actions that lead to success
    last_action_text: str  # TextWorld action representation for the last action
    original_system_prompt: str | None = None
    
    # TextWorld shared GameState attributes
    tw_feedback: str
    tw_objective: str
    tw_max_score: int
    tw_extra_walkthrough: list[str] # same as action_trajectory, but not parsed as LLM tool call
    tw_description: str
    tw_inventory: Any
    tw_score: int
    tw_moves: int
    tw_won: bool
    tw_lost: bool


class TextWorldStepResult(EnvironmentStepResult):
    state: TextWorldState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


class TextWorldEnv(Environment):
    """
    TextWorld environments implemented as our base Environment subclasses
    """

    def __init__(
        self,
        dataset_config: dict[str, Any],
        task: str,  # one of TEXTWORLD_TASKS
        state_keys: list[str] | None = None,
        # Inherited arguments
        num_train_samples: int = 20,
        num_val_samples: int | None = None,
        num_test_samples: int = 5,  # test_samples maybe the same as val_samples
        max_turns: int | None = None,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = "You are a helpful tool-calling assistant.",
        **kwargs: Any,
    ) -> None:
        # max_turns in base class is used for generic envs; TextWorld uses `max_steps`
        # but we still set base max_turns to something sensible.
        super().__init__(seed=seed, split=split, **kwargs)

        self.textworld_games_path = dataset_config["cache_dir"]
        self.task = task
        self.state_keys = state_keys or ["description", "score", "moves"]
        self.all_tasks = list(DEFAULT_MAX_TURNS_BY_TASK.keys())
        
        # Build environment
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.split = split

        self.max_turns = max_turns or DEFAULT_MAX_TURNS_BY_TASK[task]
        self.system_prompt = system_prompt  # This gets updated based on the TextWorld task
        
        # Singleton factory shared across env instances (so scanning happens once)
        # -> Can then create new environments via
        #    sample_env = self.factory.make_env(task=self.task, sample_id=sample_id)
        self.factory = TextWorldFactory(
            textworld_games_path=str(self.textworld_games_path),
            tasks=self.all_tasks,
            # Required infos
            objective=True,
            description=True,
            score=True,
            max_score=True,
            won=True,
        )
        if self.task not in self.factory.list_tasks():  # check that this aligns with self.all_tasks
            raise KeyError(f"Task '{self.task}' not found. Available: {self.factory.list_tasks()}")
        
        # Load data (tasks) and get splits
        self.datasets = self.init_data()

        # Initialize tools
        self.textworld_tool = TextWorldTool(task=self.task)
        self.tool_descriptions: list[dict[str, Any]] = self.textworld_tool.get_tool_descs()
        self.get_llm_toolcall_from_tw_text = self.textworld_tool.get_llm_toolcall_from_tw_text
        # -> See self._step_impl() for how we call tools

        # Few-shot prompts for Act-PRM generation + training
        self.act_prm_fewshot_prompts = ACT_PRM_FEWSHOT_PROMPTS

    def __len__(self) -> int:
        """
        Get the number of sample tasks available for the given split and task
        """
        return len(self.datasets[self.split])

    def init_data(self) -> dict[str, list[int]]:
        """
        Using the TextWorldFactory, we get the sample indices for the task corresponding to the different splits
        """
        num_tasks = self.factory.num_games(self.task)
        self._check_num_tasks(num_tasks, {
            "train": self.num_train_samples,
            "test": self.num_test_samples,
        })
        train_indices = list(range(self.num_train_samples))
        test_indices  = list(range(self.num_train_samples, self.num_train_samples + self.num_test_samples))
        eval_indices  = test_indices
        # Update val_indices if num_val_samples is provided
        if self.num_val_samples is not None:
            self._check_num_tasks(num_tasks, {
                "train": self.num_train_samples,
                "eval": self.num_val_samples,
                "test": self.num_test_samples,
            })
            eval_indices = list(range(len(test_indices), len(test_indices) + self.num_val_samples))
            
        return {
            "train": train_indices,
            "eval":  eval_indices,
            "test":  test_indices,
        }

    def _check_num_tasks(self, num_tasks: int, num_tasks_by_split: dict[str, int]) -> None:
        """
        Check that the number of tasks is sufficient for the given splits
        """
        total_split_tasks = sum(num_tasks_by_split.values())
        assert num_tasks >= total_split_tasks, (
            "Not enough tasks to split into the given splits"
            f"\n-> num_tasks_by_split: {num_tasks_by_split}"
            f"\n-> total_split_tasks: {total_split_tasks}"
            f"\n-> number of available '{self.task}' tasks: {num_tasks}"
        )

    def _build_system_prompt(self, system_prompt: str | None = None) -> str:
        # put task instructions into system prompt to keep user obs clean
        system_prompt = system_prompt or self.system_prompt
        instr = get_instruction_prompt(self.task, max_turns=self.max_turns)
        return f"{system_prompt}\n\n{instr}"

    def _key(self, key: str, key_map: dict[str, str] | None = None) -> str:
        """
        Change a TextWorld GameState key name
        """
        key_map = key_map or {"description": "current_setting"}
        return key_map.get(key, key)

    def reset(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
        try_step: int = 0,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> TextWorldState:
        """
        Reset environment (starting new episode + loading a new task)
        """
        sample_idx_adj = sample_idx % len(self.datasets[self.split])  # wrap around if out of bounds
        sample_idx_tw = self.datasets[self.split][sample_idx_adj]

        # Create and start TextWorld environment for this sample_idx
        sample_tw_env = self.factory.make_env(task=self.task, sample_id=sample_idx_tw)
        tw_game_state: GameState = sample_tw_env.reset()
        
        # Conveniently, we're provided a list of actions that lead to success
        action_trajectory: list[str] = [
            self.get_llm_toolcall_from_tw_text(tw_action)
            for tw_action in tw_game_state["extra.walkthrough"]
        ]
        
        # Prepare initial state message
        state_json = {self._key(k): v for k, v in tw_game_state.items() if k in self.state_keys}
        state_json["moves_left"] = self.max_turns - state_json["moves"]
        state_content = MESSAGE_TEMPLATE.format(
            action_feedback=tw_game_state["feedback"].strip(),
            state_json=json.dumps(state_json, indent=2),
        )
        messages = [{"role": "user", "content": state_content}]
        # Pass other attributes to our TextWorldState
        tw_game_state_kwargs = {
            f"tw_{k}".replace(".", "_"): v for
            k, v in tw_game_state.items() if k in SHARED_GAMESTATE_KEYS
        }
        # Display the ground-truth action trajectory
        if sample_idx == 0 and generation_idx == 0 and try_step == 0:
            for _ix, _tw_action in enumerate(tw_game_state_kwargs["tw_extra_walkthrough"]):
                rich_print(
                    f"{_ix}: [bright_cyan]{_tw_action}[/bright_cyan] ->"
                    f"\n[bold]{action_trajectory[_ix]}[/bold]\n{"-" * 100}"
                )
        return TextWorldState(
            system_prompt=self._build_system_prompt(),
            new_messages=messages,
            model_response=None,
            prior_messages=[],
            tools=self.tool_descriptions,
            # TextWorld-specific fields
            action_trajectory=action_trajectory,
            sample_tw_env=sample_tw_env,
            last_action_text="",
            **tw_game_state_kwargs,
            # Step-wise metadata
            sample_id=sample_idx,
            generation_id=generation_idx,
            batch_id=batch_idx,
            try_step=try_step,
            timestep=0,
            # Track for accuracy eval
            metadata={"correct": 0, "total": 1},  # correct := score == max_score
            # Past observations to show
            first_obs_to_show=len(messages) + 1,  # system + default context + user message
        )

    def step(self, **kwargs: Any) -> TextWorldStepResult:
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: TextWorldState,
        current_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> TextWorldStepResult:
        """
        Step through the environment
        """
        done = False
        truncated = False
        reward = 0.0

        # Load current task's TextWorld env
        sample_tw_env = current_state.sample_tw_env

        metadata = copy(current_state.metadata)
        timestep = int(current_state.timestep)
        try_step = int(current_state.try_step)
        updated_try_step = False

        # Create environment response
        env_messages = []
        stdout = ""
        tool_call_error = False
        # Initialize next state's game_state attributes as the current state's
        next_state_tw_game_state_kwargs = {
            k: v for k, v in vars(current_state).items() if k.startswith("tw_")
        }  # -> Will update if we take a successful step
        # Keep track of current score to compute reward
        # -> reward = tw_reward == next_game_state["score"] - current_score
        current_score = current_state.tw_score
        last_action_text = ""  # only register if we have a successful step

        # Parse actions (messages and tool calls)
        made_tool_call = False
        for action_idx, action in enumerate(parsed_actions):
            if action.type == "function_call":  # handle tool call
                fc_name = action.name
                fc_args = action.arguments
                try:  
                    # For TextWorld, we convert the LLM function call to TextWorld action text
                    tw_action_text = self.textworld_tool(tool_name=fc_name, tool_args=fc_args)
                    assert "tool call error" not in tw_action_text.lower(), (
                        f"Invalid tool call: {action.text}\n\n{tw_action_text}"
                    )
                except Exception as e:
                    # Handle a tool call error by sending this error to the LLM
                    _error_class = type(e).__name__
                    stdout = f"Invalid tool call:\n\n{action.text}\n\n{_error_class}: {e}"
                    tool_call_error = True
                    logger.warning(f"Last message not a valid tool call: {action.text}")

                if not tool_call_error:
                    # Send parsed action to TextWorld environment and take next step
                    tw_game_state, tw_reward, tw_done = sample_tw_env.step(tw_action_text)
                    assert tw_reward == tw_game_state["score"], (  # Check our understanding
                        f"Reward {tw_reward} != score {tw_game_state["score"]}"
                    )
                    # Prepare next state message
                    last_action_text = tw_action_text
                    state_json = {
                        self._key(k): v for k, v in tw_game_state.items() if k in self.state_keys
                    }
                    state_json["moves_left"] = self.max_turns - state_json["moves"]
                    state_json["last_action"] = last_action_text
                    state_content = MESSAGE_TEMPLATE.format(
                        action_feedback=tw_game_state["feedback"].strip(),
                        state_json=json.dumps(state_json, indent=2),
                    )
                    # Update GameState attributes for our TextWorldState
                    next_state_tw_game_state_kwargs.update({
                        f"tw_{k}".replace(".", "_"): v for
                        k, v in tw_game_state.items() if k in SHARED_GAMESTATE_KEYS
                    })
                    stdout = state_content
                    made_tool_call = True

                    reward = float(tw_reward - current_score)
                    done = bool(tw_done)  # simple alias
                    if tw_reward >= tw_game_state["max_score"]:
                        metadata["correct"] = 1
                    
                
                env_response = {
                    "role": "tool",
                    "type": "function_call_output",
                    "call_id": action.call_id,
                    "output": stdout,
                }
                env_messages.append(env_response)

                if tool_call_error:  # Treat as workflow failure
                    done = True
                    truncated = True
                    if not updated_try_step:
                        try_step += 1
                        updated_try_step = True
            
            elif action.type in ["message", "reasoning"]:
                # Basically ignore
                if action_idx + 1 == len(parsed_actions):  # last message
                    # # Allow model to continue with task
                    # user_content = (
                    #     "Ok please continue! Remember you *must* call tools to complete the task."
                    # )
                    # env_messages.append({"role": "user", "content": user_content})
                    
                    # Treat as workflow failure
                    user_content = "Sad! You must persist and call tools to complete the task."
                    done = True
                    truncated = True
                    env_messages.append({"role": "user", "content": user_content})
                    if not updated_try_step:
                        try_step += 1
                        updated_try_step = True
                    logger.warning(f"Last message not a tool call: {action.text}")
            else:
                logger.error(f"Invalid parsed actions: {parsed_actions}")
                logger.error(f"Specific unknown action type for action {action_idx}: {action.type}")
                breakpoint()

        # Update timesteps, fail if too many turns
        timestep = timestep + 1
        if timestep >= self.max_turns + 1:  # some leeway?
            truncated = True
            done = True
            env_messages.append({"role": "user", "content": self.truncation_message})
            if not updated_try_step:
                try_step += 1
                updated_try_step = True

        # Handle badness (environment should always respond to LLM response)
        if len(env_messages) == 0:
            env_messages.append({
                "role": "user",
                "content": "No tool calls were parsed. Please try again",
            })
            rich_print(f"Last action: {action.text}")
            rich_print("\n".join(
                f"Action {_act_idx}: {_action}" for _act_idx, _action in enumerate(parsed_actions)
            ))
            logger.error("No tool calls were parsed.")

        # Handle past observations to show
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
        new_state = TextWorldState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages,
            tools=current_state.tools,
            # TextWorld-specific fields
            action_trajectory=current_state.action_trajectory,
            sample_tw_env=sample_tw_env,
            last_action_text=last_action_text,
            **next_state_tw_game_state_kwargs,
            # Step-wise metadata
            sample_id=current_state.sample_id,
            generation_id=current_state.generation_id,
            batch_id=current_state.batch_id,
            try_step=try_step,
            timestep=timestep,
            # Track for accuracy eval
            metadata=metadata,
            # Past observations to show
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )
        return TextWorldStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=new_state.metadata,  # alternative access
        )


class AsyncTextWorldEnv(TextWorldEnv):
    """
    Asynchronous TextWorld environment
    """
    
    async def reset_async(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
        try_step: int = 0,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> TextWorldState:
        """
        Asynchronous reset -> assumes super().reset() is fast and non-blocking
        """
        return super().reset(
            sample_idx=sample_idx,
            generation_idx=generation_idx,
            try_step=try_step,
            batch_idx=batch_idx,
            **kwargs,
        )

    async def step_async(self, **kwargs: Any) -> TextWorldStepResult:
        """
        Asynchronous step -> assumes super().step() is fast and non-blocking
        """
        return super().step(**kwargs)
