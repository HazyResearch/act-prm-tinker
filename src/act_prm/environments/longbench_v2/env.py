"""
LongBench environment
"""

import logging
from copy import copy
from os.path import join
from typing import Any

import numpy as np
from datasets import DatasetDict, load_dataset
from pydantic import InstanceOf
from rich import print as rich_print
from transformers import AutoTokenizer

# from act_prm.environments.browsecomp_plus.tools import SearchTool, ScrollUpTool, ScrollDownTool

from ...llm_handlers import ActionFromLLM
from ..base import BaseTool, Environment
from ..types import EnvironmentStateWithAnswer, EnvironmentStepResult
from .prompts import INITIAL_PROMPT_TEMPLATE
from .tools import SearchTool, ScrollUpTool, ScrollDownTool
from .utils import chunk_text_by_tokens, convert_text_chunks_to_dicts

logger = logging.getLogger(__name__)


RESULT_TEMPLATE = """## Document View:
'''
{document}
'''{scroll_message}
"""

class LongBenchState(EnvironmentStateWithAnswer):
    """
    State of the LongBench environment
    """
    tool_registry: dict[str, InstanceOf[BaseTool]]  # callable tools, by name
    tools: list[dict[str, Any]]                     # tool descriptions
    doc_dict: dict[str, Any] | None
    all_doc_dicts: list[dict[str, Any]]
    current_doc_id: int


class LongBenchStepResult(EnvironmentStepResult):
    """
    Step result of the LongBench environment
    """
    state: LongBenchState


class LongBenchEnvironment(Environment):
    """
    LongBench environment
    """
    def __init__(
        self,
        dataset_config: dict[str, Any],
        search_tool_config: dict[str, Any],
        hf_repo_id: str = "longbench_v2-search",
        max_preview_tokens: int = 204,  # about 1024 tokens overall
        doc_chunk_size: int = 1024,
        doc_chunk_overlap: int = 128,
        num_train_samples: int = 750,
        num_val_samples: int = 30,
        num_test_samples: int = 50,
        max_turns: int = 20,
        num_tries: int = 1,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = "You are a helpful assistant that can answer questions and call tools.",
        num_fewshot_prompts: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_turns=max_turns, num_tries=num_tries, seed=seed, **kwargs)
        self.dataset_config = dataset_config
        self.hf_repo_id = hf_repo_id

        # Search tool config
        self.search_tool_config = search_tool_config

        # Build environment
        self.max_preview_tokens = max_preview_tokens
        self.doc_chunk_size = doc_chunk_size
        self.doc_chunk_overlap = doc_chunk_overlap

        # Initialize default context (fewshot prompts) for all samples
        self.num_fewshot_prompts = num_fewshot_prompts
        self.default_context = self.get_default_context()
        
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.split = split

        self.max_turns = max_turns
        self.num_tries = num_tries
        self.seed = seed
        self.split = split

        # Load data and tools
        self.system_prompt = system_prompt
        self.system_prompt_message = {"role": "system", "content": self.system_prompt}
        self.datasets = self.init_data()

        # # Initialize tools -> maybe should be done per reset call?
        # self.tool_registry = {
        #     "search": SearchTool(**search_tool_config),
        # }
    
        # From BrowseComp-Plus
        # self.datasets, self.ds_corpus = self.init_data()
        # # Build index based on doc_id for self.ds_corpus
        # self.ds_corpus_index = {v: i for i, v in enumerate(self.ds_corpus["doc_id"])}
        # # Initialize tools
        # _tool_kwargs = {
        #     "doc_dataset": self.ds_corpus,
        #     "ds_corpus_index": self.ds_corpus_index,
        # }
        # self.tool_registry = {
        #     "scroll_up": ScrollUpTool(**_tool_kwargs),
        #     "scroll_down": ScrollDownTool(**_tool_kwargs),
        #     "search": SearchTool(
        #         corpus=self.ds_corpus,
        #         tokenizer=self.tokenizer,  # Inherited; see _init_tokenizer(self) in ../base.py
        #         max_preview_tokens=self.max_preview_tokens,
        #         **self.search_tool_config,
        #     ),
        # }
    
    def get_default_context(self) -> list[dict[str, str]]:
        """
        Build fewshot examples, i.e., default context, for all samples
        """
        return []  # MZ 1/19/2026: We can figure out if we need few-shot examples

    def init_data(self) -> DatasetDict:
        """
        Load raw data from HF dataset hub and split into train/val/test
        """
        ds = load_dataset(**self.dataset_config)
        trainval_test_dict = ds.train_test_split(
            test_size=self.num_test_samples, shuffle=True, seed=self.seed
        )
        train_val_dict = trainval_test_dict["train"].train_test_split(
            test_size=self.num_val_samples, shuffle=True, seed=self.seed
        )
        return DatasetDict({
            "train": train_val_dict["train"],
            "eval": train_val_dict["test"],
            "test": trainval_test_dict["test"],
        })

    def shuffle(self, seed: int | None = None) -> None:
        """
        Shuffle dataset
        """
        seed = seed or self.seed
        np.random.seed(seed)
        indices = np.arange(len(self.datasets[self.split]))
        np.random.shuffle(indices)
        self.datasets[self.split] = self.datasets[self.split][indices]
        
    def reset(
        self,
        sample_idx: int,
        generation_idx: int,
        try_step: int = 0,
        batch_idx: int = 0,
    ) -> LongBenchState:
        """
        Reset environment (starting new episode + loading a new task)
        """
        sample_idx_adj = self.adjust_sample_idx(sample_idx)  # Wrap around if out of bounds
        sample = self.datasets[self.split][sample_idx_adj]
        document = sample["context"]
        # Split document into chunks
        _, text_chunks = chunk_text_by_tokens(document, self.tokenizer)
        all_doc_dicts = convert_text_chunks_to_dicts(text_chunks)
        current_chunk_id = 0  # start with the first chunk
        doc_dict = all_doc_dicts[current_chunk_id]
        # Get answer choices
        choices = []
        for k in ["choice_A", "choice_B", "choice_C", "choice_D"]:
            _letter = k[len("choice_"):].upper()
            choices.append(f"{_letter}: {sample[k]}")
        choices = "\n".join(choices)

        # Build prompt and save answer
        question = sample["question"]
        answer = sample["answer"].upper()  # convert to uppercase letter for matching
        prompt = INITIAL_PROMPT_TEMPLATE.format(
            question=question,
            choices=choices,
            document=doc_dict["text"],
        )
        _search_save_path = join(
            self.dataset_config["cache_dir"],
            f"longbench_v2-{self.split}-{sample_idx:04d}"
        )
        tool_registry = {
            "search": SearchTool(
                corpus=all_doc_dicts,
                save_path=_search_save_path,
                tokenizer=self.tokenizer,  # Inherited; see _init_tokenizer(self) in ../base.py
                **self.search_tool_config,
            ),
            "scroll_down": ScrollDownTool(doc_dict=doc_dict),
            "scroll_up": ScrollUpTool(doc_dict=doc_dict),
        }
        tools = [tool.get_tool_desc() for tool in tool_registry.values()]
        messages = [{"role": "user", "content": prompt}]

        return LongBenchState(
            system_prompt=self.system_prompt,
            new_messages=messages,
            model_response=None,
            prior_messages=[],
            tool_registry=tool_registry,
            tools=tools,
            # LongBench-specific fields
            question=question,
            answer=answer,
            doc_dict=doc_dict,
            all_doc_dicts=all_doc_dicts,
            current_doc_id=current_chunk_id,
            # Step-wise metadata
            sample_id=sample_idx,
            generation_id=generation_idx,
            batch_id=batch_idx,
            try_step=try_step,
            timestep=0,
            # Track for accuracy eval
            metadata={"correct": 0, "total": 1},
            # Past observations to show
            first_obs_to_show=len(messages) + 1,  # system + default context + user message
        )

    def step(self, **kwargs: Any) -> LongBenchStepResult:
        """
        Step through the environment; see `_step_impl` for details
        """
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: LongBenchState,
        current_messages: list[dict[str, Any]] | None = None,
    ) -> LongBenchStepResult:
        """
        Step through the environment
        """
        question = str(current_state.question)
        answer   = str(current_state.answer)

        scroll_tool_kwargs = {
            "all_doc_dicts": current_state.all_doc_dicts,
            "current_doc_id": current_state.current_doc_id,
            "doc_dict": current_state.doc_dict,
        }
        available_tool_names = [k for k in current_state.tool_registry.keys()]
        
        done = False
        truncated = False
        reward = 0
        updated_try_step = False
        
        metadata = copy(current_state.metadata)
        timestep = copy(current_state.timestep)
        try_step = copy(current_state.try_step)
        sample_id = current_state.sample_id
        generation_id = current_state.generation_id
        batch_id = current_state.batch_id

        # Create environment response
        env_messages = []
        # Keep track of next doc_id and doc
        # -> Tool call should update, but by default will be same as current state
        next_doc_id = copy(current_state.current_doc_id)
        next_doc_dict = copy(current_state.doc_dict)

        # Parse actions (messages and tool calls)
        made_tool_call = False
        for action_idx, action in enumerate(parsed_actions):
            if action.type == "function_call":  # handle tool call
                fc_name = action.name
                fc_args = action.arguments

                if fc_name == "invalid_tool_call":
                    stdout = f"Invalid tool call:\n\n{action.text}"

                elif fc_name not in current_state.tool_registry:
                    stdout = (
                        f"Invalid tool call:\n\n{action.text}\n\n"
                        f"'{fc_name}' not currently available."
                    )
                
                else:
                    try:  # Execute tool call
                        tool = current_state.tool_registry[fc_name]
                        new_doc_dict, maybe_result_str = tool(**fc_args, **scroll_tool_kwargs)
                        # maybe_doc_result = result_str
                    except Exception as e:
                        # Handle a tool call error by sending this error to the LLM
                        _error_class = type(e).__name__
                        new_doc_dict = None
                        maybe_result_str = (
                            f"Invalid tool call:\n\n{action.text}\n\n{_error_class}: {e}"
                        )
                        # maybe_result_str = str(e)
                        # breakpoint()

                    if new_doc_dict is not None:
                        scroll_msg = ""
                        # scroll_down_msg = "\n- Scroll down for more..."
                        # scroll_up_msg   = "\n- Scroll up for more..."
                        available_tool_names = ["search"]
                        try:
                            if (
                                new_doc_dict["next_chunk_idx"] is not None
                                # and scroll_down_msg not in scroll_msg
                            ):
                                # scroll_msg += scroll_down_msg
                                available_tool_names.append("scroll_down")
                            if (
                                new_doc_dict["prev_chunk_idx"] is not None
                                # and scroll_up_msg not in scroll_msg
                            ):
                                # scroll_msg += scroll_up_msg
                                available_tool_names.append("scroll_up")
                            
                            stdout = RESULT_TEMPLATE.format(
                                document=new_doc_dict["text"],
                                scroll_message="",  # scroll_msg,
                            )
                            # Update the document and identifiers
                            if isinstance(new_doc_dict, dict):
                                next_doc_id = new_doc_dict["chunk_idx"]
                                next_doc_dict = new_doc_dict
                                made_tool_call = True
                        
                        except Exception as e:
                            _error_class = type(e).__name__
                            logger.error(f"{_error_class}: {e}")
                            print(new_doc_dict.keys())
                            breakpoint()
                    else:
                        stdout = maybe_result_str

                env_response = {
                    "role": "tool",
                    "type": "function_call_output",
                    "call_id": action.call_id,
                    "output": stdout,  # or "content" for HF transformers
                }
                env_messages.append(env_response)

            elif action.type in ["message", "reasoning"]:
                text = action.text or ""
                if (
                    action.type == "message"
                    and action_idx + 1 == len(parsed_actions)
                ):
                    done = True
                    if "Final Answer: " in text:  # Last action was an answer submission
                        ans_pred = text.split("Final Answer: ")[-1].strip().lower()
                        ans_true = answer.lower()
                        reward = float(ans_pred == ans_true)  # convert bool to float for rewar
                        if reward == 1:
                            user_content = "# RESULT: CORRECT!"
                        else:
                            user_content = "# RESULT: INCORRECT!"
                        metadata["correct"] = reward
                        metadata["total"] = 1
                    else:
                        # Allow model to continue with task
                        user_content = (
                            "Ok! Please continue with the task. Remember when you're ready to"
                            " answer, put your response as one letter in the following format:"
                            "\n\nFinal Answer: <your chosen answer letter (A, B, C, or D)>"
                        )
                    env_messages.append({"role": "user", "content": user_content})
            else:
                logger.error(f"Invalid parsed actions: {parsed_actions}")
                logger.error(f"Specific unknown action type for action {action_idx}: {action.type}")
                breakpoint()

        # Update timesteps, fail if too many turns
        timestep = timestep + 1
        if timestep >= self.max_turns:
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
                "content": "No tool calls or final answers were parsed. Please try again",
            })
            rich_print(f"Last action: {action.text}")
            for _act_idx, _action in enumerate(parsed_actions):
                rich_print(f"Action {_act_idx}: {_action}")
            logger.error("No tool calls or final answers were parsed.")

        # Let model see available tools
        available_tools_str = "\n".join(f"- {_name}" for _name in available_tool_names)
        available_tools_str = f"# Currently Available Tools:\n{available_tools_str}"
        _content_key = "output" if "output" in env_messages[-1] else "content"
        env_messages[-1][_content_key] += f"\n\n{available_tools_str}"

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
        new_state = LongBenchState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages,
            tool_registry=current_state.tool_registry,
            tools=current_state.tools,
            # LongBench-specific fields
            question=question,
            answer=answer,
            doc_dict=next_doc_dict,
            all_doc_dicts=current_state.all_doc_dicts,
            current_doc_id=next_doc_id,
            # Step-wise metadata
            sample_id=sample_id,
            generation_id=generation_id,
            batch_id=batch_id,
            try_step=try_step,
            timestep=timestep,
            # Track for accuracy eval
            metadata=metadata,
            # Past observations to show
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )
        return LongBenchStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=new_state.metadata,  # alternative access
        )


class AsyncLongBenchEnvironment(LongBenchEnvironment):
    """
    Asynchronous LongBench environment
    """
    async def reset_async(self, **kwargs: Any) -> LongBenchState:
        """
        Asynchronous reset -> assumes super().reset() is fast and non-blocking
        """
        return super().reset(**kwargs)

    async def step_async(self, **kwargs: Any) -> LongBenchStepResult:
        """
        Asynchronous step -> assumes super().step() is fast and non-blocking
        """
        return super().step(**kwargs)
