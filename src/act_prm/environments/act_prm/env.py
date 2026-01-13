"""
Action Process Reward Environments

We train LLMs to generate thoughts that lead to observed actions with high-likelihoods,
where reward = p(observed_action | thought).

We do this via two environments.
1. `ActionFirstProcessRewardEnv`: To bootstrap generations, we first give the ground-truth
   action to the LLM first, and ask it to generate the thought that leads to that action.

2. `ActionProcessRewardEnv`: We can then optionally train the LLM to generate thoughts
    based on just the present state (i.e., it does not see the action as a hint).
"""

from typing import Any

import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm

from ...llm_handlers import ActionFromLLM
from ..base import Environment
from ..types import EnvironmentState

from .utils import get_chat_from_ds_sample, get_latent_completion, get_thought_and_actions


SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful assistant that infers latent thoughts behind observed actions."
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
    assistant_indices: list[int]
    generated_thoughts: list[str]
    action_target: str



class ActionFirstProcessRewardEnv(Environment):
    """
    ActPRM environment where we prompt LLMs with ground-truth actions
    to generate thoughts that lead to that action.
    """

    def __init__(
        self,
        dataset_config: dict[str, Any],
        success_rollouts_only: bool = True,
        num_fewshot_prompts: int = 1,
        action_bos: str = "<tool_call>",
        action_eos: str = "</tool_call>",
        final_answer_bos: str = "Final Answer: ",
        num_train_samples: int = 1000,
        num_val_samples: int = 64,
        num_test_samples: int = 100,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = SYSTEM_PROMPT["content"],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_config = dataset_config
        self.success_rollouts_only = success_rollouts_only
        self.num_fewshot_prompts = num_fewshot_prompts
        
        # Use for parsing thoughts and actions from LLM messages
        self.thought_action_kwargs = {
            "action_bos": action_bos,
            "action_eos": action_eos,
            "final_answer_bos": final_answer_bos,
        }
        # Build fewshot examples, i.e., default context, for all samples
        fewshot_prompts = []
        for fewshot_prompt in THOUGHT_ACTION_FEWSHOT_PROMPTS[:self.num_fewshot_prompts]:
            fewshot_prompts.extend(fewshot_prompt)
        if len(fewshot_prompts) > 0:
            self.default_context, _ = get_latent_completion(
                fewshot_prompts, continue_final_message=False, **self.thought_action_kwargs
            )
        else:
            self.default_context = []
        
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

        self.seed = seed
        self.split = split

        self.system_prompt = system_prompt
        self.datasets = self.init_data()

    def init_data(self) -> dict[str, list[Any]]:
        """
        Initialize thought-action datasets from given process logs
        """
        ds = load_dataset(**self.dataset_config)
        if "is_train" in ds.column_names:
            # Hack, but ensures we only have train samples
            ds = ds.filter(lambda x: x["is_train"])
        if self.success_rollouts_only:
            # Filter for samples only belonging to successful rollouts
            # -> Below works as final_reward = 1 if success, -1 otherwise,
            #    all other rewards are 0
            ds = ds.filter(lambda x: x["return_"] > 0)

        # Organize source samples into Act-PRM samples
        # -> Each Act-PRM sample is a single trajectory
        unique_sample_ids = ds.unique("data_sample_id")
        all_trajectories = []
        current_trajectory = []
        current_sample_id_ = unique_sample_ids[0]
        for sample in tqdm(ds):
            if sample["data_sample_id"] == current_sample_id_:
                current_trajectory.append(sample)
            else:
                all_trajectories.append(current_trajectory)
                current_trajectory = [sample]
                current_sample_id_ = sample["data_sample_id"]
        all_trajectories.append(current_trajectory)  # add last trajectory

        # Organize into dataset splits
        num_samples = min(
            len(all_trajectories),
            self.num_train_samples + self.num_val_samples + self.num_test_samples,
        )
        shuffle_indices = list(range(num_samples))
        np.random.seed(self.seed)
        np.random.shuffle(shuffle_indices)
        last_eval_idx = self.num_train_samples + self.num_val_samples
        train_indices = shuffle_indices[:self.num_train_samples]
        eval_indices  = shuffle_indices[self.num_train_samples:last_eval_idx]
        test_indices  = shuffle_indices[last_eval_idx:]
        datasets = {
            "train": [all_trajectories[i] for i in train_indices],
            "eval":  [all_trajectories[i] for i in eval_indices],
            "test":  [all_trajectories[i] for i in test_indices],
        }
        return datasets

    def reset(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
        try_step: int = 0,
        batch_idx: int = 0,
    ) -> EnvironmentStateWithAnswer:
        """
        Reset environment (starting new episode + loading a new task)
        """
        sample_idx_adj = self.adjust_sample_idx(sample_idx)  # Wrap around if out of bounds
        sample_trajectory = self.datasets[self.split][sample_idx_adj]
        # Initial sample is just first step of the trajectory
        sample = sample_trajectory[0]
        messages, _ = get_chat_from_ds_sample(sample)
        # Get actions only from messages
        messages = [
            get_thought_and_actions(msg, **self.thought_action_kwargs)[1] for msg in messages
        ]
        # Remove system prompt in messages
        messages = messages[1:] if messages[0]["role"] == "system" else messages
        assistant_indices = [i for i, msg in enumerate(chat) if msg["role"] == "assistant"]
        # Get prompt for thought generation, and target action (str)
        latent_chat, action_str = get_latent_completion(chat, continue_final_message=True)
        # latent_chat = [self.system_prompt] + self.default_context + latent_chat
        latent_chat = self.default_context + latent_chat
        # tokens_state_action_next_obs = self.tokenizer.apply_chat_template(
        #     latent_chat,
        #     tokenize=True,
        #     add_generation_prompt=False,
        #     continue_final_message=True,  # remove added eos_token
        # )
        return EnvironmentStateWithAnswer(
            system_prompt=self.system_prompt,
            new_messages=latent_chat,
            model_response=None,
            prior_messages=[],
            tools=[],  # leave out for now
            # Step-wise metadata
            sample_id=sample_idx,
            generation_id=generation_idx,
            batch_id=batch_idx,
            try_step=try_step,
            timestep=0,
        )

        
        
        
        chat, (reward, return_) = get_chat_from_ds_sample(sample)
        return EnvironmentStateWithAnswer(
            system_prompt=self.system_prompt,

        

