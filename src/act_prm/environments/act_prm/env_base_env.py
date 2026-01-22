"""
Action Process Reward Environments

We train LLMs to generate thoughts that lead to observed actions with high-likelihoods,
where reward = p(observed_action | thought).

Based on the Generator, we do this via two setups:

1. `TinkerActionPromptActPrmGenerator`: To bootstrap generations, we first give the ground-truth
   action to the LLM first, and ask it to generate the thought that leads to that action.
   
2. `TinkerActPrmGenerator`: We can then optionally train the LLM to generate thoughts
    based on just the present state (i.e., it does not see the action as a hint).
"""

from copy import copy, deepcopy
from typing import Any

from ...llm_handlers import ActionFromLLM, get_actions
from ..base import Environment
from ..types import EnvironmentState, EnvironmentStepResult

SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a helpful assistant that infers reasoning thoughts behind observed actions."
}

THOUGHT_ACTION_FEWSHOT_PROMPTS = [
    [{'role': 'user', 'content': "## Instruction\nGiven a list of titles, think and call tools to answer this question:\n'''\nWhich documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\n'''\n\nYou may only visit the titles provided. Only call the `visit` tool once per turn.\n\nYour final answer should be a concise sentence, in the following format: 'Final Answer: <put your answer here>'.\n\n## Tool Calling\nYou can only search the following titles:\n\n- 'Adam (musical)'\n- 'Adam Clayton Powell (film)'\n- 'Adam Clayton Powell Jr.'\n- 'Adam Clayton Powell IV'\n- 'Seventh Avenue (Manhattan)'\n- 'Mother African Methodist Episcopal Zion Church'\n- 'Abyssinian Baptist Church'\n- 'Adam Clayton Powell Jr. State Office Building'\n- 'The Saimaa Gesture'\n- 'Adam Clayton Powell Sr.'\n\n## Instruction (again)\nNow answer the original question. Recall the question is:\n'''\nWhich documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\n'''\n\nVERY IMPORTANT: You may only use the provided `visit` tool once per turn, and only use the given titles to answer this question. If you provide a title not in the given titles, the tool call will fail."}, {'role': 'assistant', 'content': 'I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.\n\n<tool_call>\n{"name": "visit", "arguments": {"title": "Adam Clayton Powell"}}\n</tool_call>'}, {'role': 'tool', 'content': "Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito']."}, {'role': 'assistant', 'content': 'To find the documentary, I can search Adam Clayton Powell (film).\n\n<tool_call>\n{"name": "visit", "arguments": {"title": "Adam Clayton Powell (film)"}}\n</tool_call>'}, {'role': 'tool', 'content': 'Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.'}, {'role': 'assistant', 'content': 'Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.\n\nFinal Answer: The Saimaa Gesture'}, {'role': 'user', 'content': '# RESULT: CORRECT!'}],
    [{'role': 'user', 'content': "## Instruction\nGiven a list of titles, think and call tools to answer this question:\n'''\nWhat profession does Nicholas Ray and Elia Kazan have in common?\n'''\n\nYou may only visit the titles provided. Only call the `visit` tool once per turn.\n\nYour final answer should be a concise sentence, in the following format: 'Final Answer: <put your answer here>'.\n\n## Tool Calling\nYou can only search the following titles:\n\n- 'Baby Doll'\n- 'Nicholas Ray'\n- 'Elia Kazan'\n- 'Art &amp; Gadg'\n- 'A Letter to Elia'\n- 'The Visitors (1972 film)'\n- 'Truckline Cafe'\n- 'All My Sons'\n- 'Maya Kazan'\n- 'Macao (film)'\n\n## Instruction (again)\nNow answer the original question. Recall the question is:\n'''\nWhat profession does Nicholas Ray and Elia Kazan have in common?\n'''\n\nVERY IMPORTANT: You may only use the provided `visit` tool once per turn, and only use the given titles to answer this question. If you provide a title not in the given titles, the tool call will fail."}, {'role': 'assistant', 'content': 'I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.\n\n<tool_call>\n{"name": "visit", "arguments": {"title": "Nicholas Ray"}}\n</tool_call>'}, {'role': 'tool', 'content': 'Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.'}, {'role': 'assistant', 'content': 'Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.\n\n<tool_call>\n{"name": "visit", "arguments": {"title": "Elia Kazan"}}\n</tool_call>'}, {'role': 'tool', 'content': 'Elia Kazan was an American film and theatre director, producer, screenwriter and actor.'}, {'role': 'assistant', 'content': 'Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.\n\nFinal Answer: director, screenwriter, actor'}, {'role': 'user', 'content': '# RESULT: CORRECT!'}],
    [{'role': 'user', 'content': "## Instruction\nGiven a list of titles, think and call tools to answer this question:\n'''\nWhich magazine was started first Arthur's Magazine or First for Women?\n'''\n\nYou may only visit the titles provided. Only call the `visit` tool once per turn.\n\nYour final answer should be a concise sentence, in the following format: 'Final Answer: <put your answer here>'.\n\n## Tool Calling\nYou can only search the following titles:\n\n- 'Radio City (Indian radio station)'\n- 'History of Albanian football'\n- 'Echosmith'\n- 'Women's colleges in the Southern United States'\n- 'First Arthur County Courthouse and Jail'\n- 'Arthur's Magazine'\n- '2014–15 Ukrainian Hockey Championship'\n- 'First for Women'\n- 'Freeway Complex Fire'\n- 'William Rast'\n\n## Instruction (again)\nNow answer the original question. Recall the question is:\n'''\nWhich magazine was started first Arthur's Magazine or First for Women?\n'''\n\nVERY IMPORTANT: You may only use the provided `visit` tool once per turn, and only use the given titles to answer this question. If you provide a title not in the given titles, the tool call will fail."}, {'role': 'assistant', 'content': 'I need to search Arthur\'s Magazine and First for Women, and find which was started first.\n\n<tool_call>\n{"name": "visit", "arguments": {"title": "Arthur\'s Magazine"}}\n</tool_call>'}, {'role': 'tool', 'content': "Arthur's Magazine (1844-\x80\x931846) was an American literary periodical published in Philadelphia in the 19th century."}, {'role': 'assistant', 'content': 'Arthur\'s Magazine was started in 1844. I need to search First for Women next.\n\n<tool_call>\n{"name": "visit", "arguments": {"title": "First for Women"}}\n</tool_call>'}, {'role': 'tool', 'content': "First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989."}, {'role': 'assistant', 'content': "First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.\n\nFinal Answer: Arthur's Magazine"}, {'role': 'user', 'content': '# RESULT: CORRECT!'}],
    [{'role': 'user', 'content': "## Instruction\nGiven a list of titles, think and call tools to answer this question:\n'''\nWere Pavel Urysohn and Leonid Levin known for the same type of work?\n'''\n\nYou may only visit the titles provided. Only call the `visit` tool once per turn.\n\nYour final answer should be a concise sentence, in the following format: 'Final Answer: <put your answer here>'.\n\n## Tool Calling\nYou can only search the following titles:\n\n- 'Leonid Ramzin'\n- 'Leftover hash lemma'\n- 'Phthinosaurus'\n- 'Kate Dillon Levin'\n- 'Tony Levin'\n- 'Gabriel Levin'\n- 'Leonid Levin'\n- 'Pavel Urysohn'\n- 'Meyer Levin'\n- 'Leonid Bunimovich'\n\n## Instruction (again)\nNow answer the original question. Recall the question is:\n'''\nWere Pavel Urysohn and Leonid Levin known for the same type of work?\n'''\n\nVERY IMPORTANT: You may only use the provided `visit` tool once per turn, and only use the given titles to answer this question. If you provide a title not in the given titles, the tool call will fail."}, {'role': 'assistant', 'content': 'I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.\n\n<tool_call>\n{"name": "visit", "arguments": {"title": "Pavel Urysohn"}}\n</tool_call>'}, {'role': 'tool', 'content': 'Pavel Samuilovich Urysohn (February 3, 1898 â\x80\x93 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.'}, {'role': 'assistant', 'content': 'Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.\n\n<tool_call>\n{"name": "visit", "arguments": {"title": "Leonid Levin"}}\n</tool_call>'}, {'role': 'tool', 'content': 'Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.'}, {'role': 'assistant', 'content': 'Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.\n\nFinal Answer: yes'}, {'role': 'user', 'content': '# RESULT: CORRECT!'}],
]


class ActionProcessRewardState(EnvironmentState):
    """
    State of the ActionProcessReward environment
    """
    action_target: str
    action_step_id: int
    base_env_state: EnvironmentState           # State of the base env that we take actions in
    action_trajectory: list[dict[str, str]]    # True action-only trajectory    


class ActionProcessRewardStepResult(EnvironmentStepResult):
    """
    Step result of the ActionProcessReward environment
    """
    state: ActionProcessRewardState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


class AsyncActPrmEnvWithBaseEnv(Environment):
    """
    ActPRM environment where we prompt LLMs with just the present state
    to generate thoughts that lead to the next observed action.
    """

    def __init__(
        self,
        dataset_config: dict[str, Any] | None,
        base_env: Environment,
        success_rollouts_only: bool = True,
        actions_only: bool = True,
        num_fewshot_prompts: int = 1,
        action_bos: str = "<tool_call>",
        action_eos: str = "</tool_call>",
        final_answer_bos: str = "Final Answer: ",
        thought_bos: str = "<thought>",
        thought_eos: str = "</thought>",
        num_train_samples: int = 100,
        num_val_samples: int = 64,
        num_test_samples: int = 10,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = SYSTEM_MESSAGE["content"],
        original_system_prompt: str | None = None,
        max_messages: int = 1000,  # a bit hacky, but truncate for long contexts
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_config = dataset_config  # Use to load sequences of actions
        self.base_env = base_env  # Original environment to get next_states from
        self.base_env.split = split
        self.max_turns = base_env.max_turns
        
        self.success_rollouts_only = success_rollouts_only
        self.actions_only = actions_only
        self.num_fewshot_prompts = num_fewshot_prompts
        self.max_messages = max_messages
        
        # Use for parsing thoughts and actions from LLM messages
        self.action_bos = action_bos
        self.action_eos = action_eos
        self.final_answer_bos = final_answer_bos
        self.thought_bos = thought_bos
        self.thought_eos = thought_eos
        self.thought_action_kwargs = {
            "action_bos": self.action_bos,
            "action_eos": self.action_eos,
            "final_answer_bos": self.final_answer_bos,
        }
        # Build fewshot examples, i.e., default context, for all samples
        self.default_context = self.get_default_context()
        
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

        self.seed = seed
        self.split = split

        self.system_prompt = system_prompt
        self.original_system_prompt = original_system_prompt
        self.datasets = self.init_data()

    def _get_action_target(self, message_dict_or_str: dict[str, str] | str) -> str:
        """
        Get the action target from a message dict or string
        """
        if isinstance(message_dict_or_str, dict):
            return message_dict_or_str["content"]
        return message_dict_or_str

    def get_default_context(self) -> list[dict[str, str]]:
        """
        Build fewshot examples, i.e., default context, for all samples
        """
        fewshot_prompts = []

        _all_thought_action_prompts = getattr(
            self.base_env, "act_prm_fewshot_prompts", THOUGHT_ACTION_FEWSHOT_PROMPTS
        )
        for fewshot_prompt in _all_thought_action_prompts[:self.num_fewshot_prompts]:
            fewshot_prompts.extend(fewshot_prompt)
        return fewshot_prompts

    def init_data(self) -> dict[str, list[Any]]:
        """
        Initialize thought-action datasets from given process logs
        """
        return self.base_env.datasets

    def reset(self, **kwargs: Any) -> ActionProcessRewardState:
        """
        Reset environment (starting new episode + loading a new task)
        """
        return self._reset_async(**kwargs)  # bad

    def step(self, **kwargs: Any) -> ActionProcessRewardStepResult:
        """
        Step through the environment; see `_step_impl` for details
        """
        return self._step_impl_async(**kwargs)  # bad

    async def reset_async(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
        try_step: int = 0,
        batch_idx: int = 0,
    ) -> ActionProcessRewardState:
        """
        Reset environment (starting new episode + loading a new task)
        """
        # First get an initial state from the base environment
        self.base_env.split = self.split
        sample_idx_adj = sample_idx % len(self.datasets[self.split])  # wrap around if out of bounds
        base_env_state: EnvironmentState = await self.base_env.reset_async(
            sample_idx=sample_idx_adj,
            generation_idx=generation_idx,
            try_step=try_step,
            batch_idx=batch_idx,
        )
        # Then get list of "ground-truth" actions that lead to sucess for the state
        action_trajectory: list[str] = base_env_state.action_trajectory
        action_trajectory: list[dict[str, str]] = [
            {"role": "assistant", "content": action}
            for action in action_trajectory
        ]
        tools: list[dict[str, Any]] = base_env_state.tools  # tool descriptions

        # Initial action -> "<tool_call> ... </tool_call>"
        action_target = self._get_action_target(action_trajectory[0])
        # Initial sample is just first (obs, action) of the trajectory, i.e.,
        # [{"role": "user", "content": "..."},
        #  {"role": "assistant", "content": <tool_call> ... </tool_call>}]
        # Currently, we present the next ground-truth action to the LLM as part of the chat,
        # but we can also do this via just `action_target`
        messages = base_env_state.new_messages + [{"role": "assistant", "content": action_target}]

        # MZ 1/18/26: Note that we add default context here, but we filter it out for
        # thought-action SFT trajectories

        # MZ 1/21/26 -> Replicate "fsc*", try few-shot convo
        if self.num_fewshot_prompts > 0:
            messages[0]["content"] = (
                f"Great! Now do the same for the next task:\n\n"
                f"## Next Task:\n\n{messages[0]['content']}"
            )
        new_messages = self.default_context + messages
        
        return ActionProcessRewardState(
            system_prompt=self.system_prompt,
            new_messages=new_messages,  # these are in standard (state, thought-action) format
            model_response=None,
            prior_messages=[],
            tools=tools,  # include for now, but also assumes tools are consistent for all steps
            # Act-PRM-specific fields
            action_target=action_target,
            action_step_id=0,
            action_trajectory=action_trajectory,
            base_env_state=base_env_state,
            # Step-wise metadata
            sample_id=sample_idx,
            generation_id=generation_idx,
            batch_id=batch_idx,
            try_step=try_step,
            timestep=0,
            # Past observations to show (account for default context, system prompt)
            first_obs_to_show=len(new_messages) + 1, # system + default context + user message
        )

    async def step_async(
        self,
        **kwargs: Any,
    ) -> ActionProcessRewardStepResult:
        """
        Step through the environment; see `_step_impl` for details
        """
        return await self._step_impl_async(**kwargs)

    async def _step_impl_async(
        self,
        parsed_actions: list[ActionFromLLM],
        current_state: ActionProcessRewardState,
        current_messages: list[dict[str, Any]] | None = None,
        reward: float = 0.0,
        **kwargs: Any,
    ) -> ActionProcessRewardStepResult:
        """
        Subclass implementation of `step`.

        We already compute rewards in the Act-PRM generator; we just pass them in here.
        """
        base_env_state = current_state.base_env_state
        action_target  = current_state.action_target
        action_step_id = copy(current_state.action_step_id)
        action_trajectory = current_state.action_trajectory

        done = False
        truncated = False

        metadata = copy(current_state.metadata)
        timestep = copy(current_state.timestep)
        try_step = copy(current_state.try_step)
        sample_id = current_state.sample_id
        generation_id = current_state.generation_id
        batch_id = current_state.batch_id

        # Update timesteps, fail if too many turns
        timestep += 1
        if timestep >= self.max_turns:
            truncated = True
            done = True
            reward = 0

        # Parse actions (messages and tool calls)
        action = parsed_actions[0]  # only consider first generated action
        if action.type == "message":
            # Parse the generated thought
            thought_text = action.text or ""
            # ^For SFT on action-only training, we'll pass in empty action.text
            thought_text = thought_text.split(self.thought_bos)[-1].strip()
            thought_text = thought_text.split(self.thought_eos)[0].strip()
            # Extract thought as text before action or final answer
            if len(thought_text.split(self.action_bos)) > 1:
                thought_text = self.action_bos.join(
                    thought_text.split(self.action_bos)[:-1]
                ).strip()
            if len(thought_text.split(self.final_answer_bos)) > 1:
                thought_text = self.final_answer_bos.join(
                    thought_text.split(self.final_answer_bos)[:-1]
                ).strip()

            # Create a new model message with {thought_text}\n\n{action_target}
            model_messages = [{
                "role": "assistant",
                "content": f"{thought_text}\n\n{action_target}".strip(),
            }]
            current_messages[-1] = model_messages[0]
            parsed_actions: list[ActionFromLLM] = get_actions(model_messages)
            base_env_current_messages = current_messages[:-1]  # exclude the last action message

            # Step through the base environment
            base_env_step_result: EnvironmentStepResult = await self.base_env.step_async(
                parsed_actions=parsed_actions,
                model_response=model_messages,
                current_state=base_env_state,
                # Sets next_state.prior_messages to current_messages
                current_messages=base_env_current_messages,
            )
            next_base_env_state = base_env_step_result.state
            # Add next_obs and next_action as new messages
            action_step_id += 1
            if action_step_id == len(action_trajectory):
                done = True
                env_messages = next_base_env_state.new_messages
            else:
                action_target = self._get_action_target(action_trajectory[action_step_id])
                env_messages: list[dict[str, str]] = (
                    next_base_env_state.new_messages
                    + [{"role": "assistant", "content": action_target}]
                )

            # Handle past observations to show
            current_messages = self.maybe_hide_observations(
                current_messages or [],
                first_obs_to_show=current_state.first_obs_to_show,
                last_obs_to_show=current_state.last_obs_to_show,
            )

            # Create environment response
            new_state = ActionProcessRewardState(
                system_prompt=self.system_prompt,
                new_messages=env_messages,
                model_response=None,
                prior_messages=current_messages,
                tools=current_state.tools,
                # Act-PRM-specific fields
                action_target=action_target,
                action_step_id=action_step_id,
                action_trajectory=action_trajectory,
                base_env_state=next_base_env_state,
                # Step-wise metadata
                sample_id=sample_id,
                generation_id=generation_id,
                batch_id=batch_id,
                try_step=try_step,
                timestep=timestep,
                # Past observations to show (account for default context)
                first_obs_to_show=current_state.first_obs_to_show,
                last_obs_to_show=current_state.last_obs_to_show,
            )

        else:
            # No interpretable thoughts generated, just ask to try again
            new_state = deepcopy(current_state)
            new_state.timestep = timestep
            
        return ActionProcessRewardStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=metadata,
        )
