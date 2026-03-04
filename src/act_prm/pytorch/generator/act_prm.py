"""
Base class for generation / rollout sampling for Hugging Face Transformers models
"""

import logging
from copy import copy, deepcopy
from typing import Any, Literal

from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from act_prm.environments import Environment, EnvironmentStepResult
from act_prm.environments.act_prm import ActionProcessRewardState

from act_prm.generator.tinker_act_prm import process_state_messages_for_metrics
from act_prm.generator.tinker_act_prompt_aprm import get_action_prompted_completion

from act_prm.llm_handlers import HuggingFaceLLM
from act_prm.llm_handlers.action_utils import get_actions
from act_prm.llm_handlers.types import ActionFromLLM
from act_prm.replay_buffer.types import TrajectoryGroup
from act_prm.utils.display import rich_print_messages

from .base import (
    get_action_logprobs_and_state_action_tokens,
    _get_trajectory_group_from_generations,
    HuggingFaceGenerator,
)


console = Console()
logger = logging.getLogger(__name__)
ROYGBIV = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"]


def get_act_logprobs_and_state_act_tokens_from_messages(
    model: PreTrainedModel,
    hf_tokenizer: PreTrainedTokenizerBase,
    state_messages: list[list[dict[str, str]]],
    state_action_messages: list[list[dict[str, str]]],
    state_continue_final_message: bool,
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Get the action logprobs and state action tokens from the state messages and state action messages
    """
    og_padding_side = copy(hf_tokenizer.padding_side)
    hf_tokenizer.padding_side = "right"
    # state_attn_masks: list[list[int]] = hf_tokenizer.apply_chat_template(
    #     state_messages,
    #     add_generation_prompt=False if state_continue_final_message else True,
    #     continue_final_message=state_continue_final_message,
    #     padding=True,
    #     tokenize=True,
    #     return_dict=True,
    # )["attention_mask"]
    # state_lens = [sum(attn_mask) for attn_mask in state_attn_masks]
    state_lens = [
        len(hf_tokenizer.apply_chat_template(
            _state_messages,
            add_generation_prompt=False if state_continue_final_message else True,
            continue_final_message=state_continue_final_message,
            padding=False,
            tokenize=True,
        )) for _state_messages in state_messages
    ]
    state_action_inputs = hf_tokenizer.apply_chat_template(
        state_action_messages,
        add_generation_prompt=False,
        continue_final_message=False,
        padding=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    state_action_logits = model(**state_action_inputs.to(model.device), use_cache=False).logits
    action_logprobs, state_action_tokens = get_action_logprobs_and_state_action_tokens(
        logits=state_action_logits,
        state_lens=state_lens,
        **state_action_inputs.to(model.device),  # input_ids, attention_mask
    )
    hf_tokenizer.padding_side = og_padding_side
    return action_logprobs, state_action_tokens


class ActionPromptActPrmGenerator(HuggingFaceGenerator):
    """
    Compute rollouts using Hugging Face Transformers models
    """
    def __init__(
        self,
        reward_method: Literal["action_probs", "em"] = "em",
        mean_center: bool = False,
        action_bos: str = "<tool_call>",
        action_eos: str = "</tool_call>",
        thought_bos: str = "<thought>",
        thought_eos: str = "</thought>",
        final_answer_bos: str = "Final Answer: ",
        samples_to_display: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(mean_center=mean_center, **kwargs)
        self.reward_method = reward_method  # How we compute rewards
        # Delimiters to parse thoughts and actions from response text
        self.action_bos  = action_bos
        self.action_eos  = action_eos
        self.thought_bos = thought_bos
        self.thought_eos = thought_eos
        self.final_answer_bos = final_answer_bos

        self.samples_to_display = samples_to_display  # for visualization

    def _get_thought_prompt(
        self,
        state_messages: list[dict[str, Any]],
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> tuple[list[dict[str, Any]], torch.Tensor]:
        """
        Get the thought prompt messages for the given state messages.
        We expect `state_messages` as a list of the form:
        ```
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
            ...
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        ]
        ```
        Here, we use `get_action_prompted_completion` to convert these messages to those that
        prompt for the thought
        """
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        og_padding_side = copy(hf_tokenizer.padding_side)
        hf_tokenizer.padding_side = "left"
        # Prompt for thoughts, e.g., of the form <action_bos>(action)</action_eos><thought_bos>
        state_messages = deepcopy(state_messages)
        state_messages, action_target = get_action_prompted_completion(
            state_messages, 
            continue_final_message=True,
            action_bos=self.action_bos,
            action_eos=self.action_eos,
            thought_bos=self.thought_bos,
            thought_eos=self.thought_eos,
            final_answer_bos=self.final_answer_bos,
        )
        input_ids: torch.Tensor = hf_tokenizer.apply_chat_template(
            state_messages,
            add_generation_prompt=False,
            continue_final_message=True,  # don't add eos_token to final message
            tokenize=True,
            # return_dict=True,  # Just return the (1, state_len) torch.Tensor 
            return_tensors="pt",
        )
        hf_tokenizer.padding_side = og_padding_side
        return state_messages, input_ids

    def _parse_thoughts(self, response_text: str) -> str:
        """
        Extract only the thought text from a response text
        """
        # Extract thoughts if explicitly tagged
        response_text = response_text.split(self.thought_bos)[-1].strip()
        response_text = response_text.split(self.thought_eos)[0].strip()
        # Extract thought as text before action or final answer
        if len(response_text.split(self.action_bos)) > 1:
            response_text = self.action_bos.join(response_text.split(self.action_bos)[:-1]).strip()
        if len(response_text.split(self.final_answer_bos)) > 1:
            response_text = self.final_answer_bos.join(
                response_text.split(self.final_answer_bos)[:-1]
            ).strip()
        return response_text

    def _compute_group_rewards(
        self,
        rewards_in_group: list[float],
        split: str = "train",
    ) -> list[float]:
        """
        Compute group rewards
        """
        # Expectation-maximization tells us to normalize by the sum of action_probs
        if self.reward_method == "em" and split == "train":
            sum_p = sum(rewards_in_group)
            rewards_in_group = [p / sum_p for p in rewards_in_group]
            # ^But only do for train split as eval num_return_sequences is 1
        return rewards_in_group

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
        Generate thought-action trajectories given observed actions in an Act-PRM environment.

        Unlike typical group-rollouts, at *each* step, we:
        1. Generate `num_return_sequences` thoughts,
        2. Compute the per-step reward for each generation,
        3. Pick the highest-reward thought to continue for the next step.

        This results in *1* full trajectory (from start to workflow completion).

        However for training, we still save each (state, action', thought', reward') tuple 
        for all `num_return_sequences` thoughts as a TrajectoryGroup. This results in returning
        `num_steps` TrajectoryGroups.

        Currently returns both samples of the form (state, thought, action) and (state, action, thought) for RL training.
        These are in a dict with keys:
        - "think_act_policy": (state, thought, action) used for SFT training (standard (state, action, next_obs) tuple)
        - "policy":           (state, action, thought) used for RL training for action-prompted generation
        """
        llm = llm or self.llm
        env = env or self.env
        cfg = cfg or self.cfg
        env.split = split  # Select task split
        discount_factor = discount_factor or self.discount_factor or cfg.discount_factor
        
        was_training = llm.model.training
        llm.model.eval()
        device = llm.model.device

        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        # Keep track of original padding side bc we'll change it multiple times below
        og_tokenizer_padding_side = copy(hf_tokenizer.padding_side)
        hf_tokenizer.padding_side = "left"  # confirm left-padding for generation

        # Generation parameters
        max_tokens = max_tokens or cfg.max_tokens
        temperature = temperature or cfg.temperature
        num_return_sequences = num_return_sequences or (
            cfg.group_size if split == "train" else cfg.eval_group_size
        )

        with torch.no_grad():
            # Initialize list of all trajectory groups to return
            # -> We return (1) (state, thought, action) and (2) (state, action, thought) trajectories
            #    (1) is used for SFT or TinkerActPrmGenerator RL
            #    (2) is used for TinkerActionPromptActPrmGenerator RL
            all_thought_act_trajectory_groups: list[TrajectoryGroup] = []
            all_act_thought_trajectory_groups: list[TrajectoryGroup] = []

            state: ActionProcessRewardState = env.reset(
                sample_idx=sample_id,
                generation_idx=0,
                try_step=try_step,
            )
            try:
                max_turns = len(state.assistant_indices)
            except AttributeError:
                max_turns = len(state.action_trajectory)
            
            done = False
            while not done:
                # Generate model responses and step through the environment
                # state_messages should be [obs, action, ..., obs, action]
                state_messages: list[dict[str, Any]] = self._get_messages_from_state(state) 
                # Prompt for thoughts (tokenizer should be left-padded, but should not matter?)
                act_prompt_state_messages, input_ids = self._get_thought_prompt(state_messages)
                # Generate `num_return_sequences` thoughts
                batch_state_input_ids = input_ids.expand(num_return_sequences, -1)
                state_input_len_left_padded = batch_state_input_ids.shape[1]
                # Generate model responses
                outputs = llm.model.generate(
                    input_ids=batch_state_input_ids.to(device),
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=hf_tokenizer.pad_token_id,
                    streamer=self.streamer if num_return_sequences == 1 else None,
                )
                # Decode and convert tokens to messages
                decoded_texts = hf_tokenizer.batch_decode(
                    outputs[:, state_input_len_left_padded:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                # Extract thoughts from decodes (e.g., remove any <thought>...</thought> tags)
                thoughts_in_group: list[str] = [self._parse_thoughts(txt) for txt in decoded_texts]
                # Compute per-step rewards for each thought
                # -> Get current state without last action, remove few-shot prompts
                first_msg_to_show = getattr(state, "first_obs_to_show", 0) - 3
                # ^-1 ActPRM environment previously counts system prompt as first message,
                # but we apply after system_prompt in process_state_messages_for_metrics
                standard_chat: list[dict[str, str]] = process_state_messages_for_metrics(
                    state_messages,
                    system_prompt=getattr(env, "original_system_prompt", state.system_prompt),
                    first_msg_to_show=max(first_msg_to_show, 0)
                )
                state_len = len(hf_tokenizer.apply_chat_template(
                    standard_chat,
                    add_generation_prompt=True,
                    continue_final_message=False,
                    tokenize=True,
                ))
                # Compute action probabilities for each thought for rewards
                # 1. First get state_thought_lens
                state_thought_messages = [
                    standard_chat + [{"role": "assistant", "content": thought}]
                    for thought in thoughts_in_group
                ]
                thought_act_messages: list[list[dict[str, str]]] = [
                    [{"role": "assistant", "content": f"{thought}\n\n{state.action_target}"}]
                    for thought in thoughts_in_group
                ]
                state_thought_act_messages: list[list[dict[str, str]]] = [
                    standard_chat + _thought_act_msgs for _thought_act_msgs in thought_act_messages
                ]
                _state_thought_act_outputs = get_act_logprobs_and_state_act_tokens_from_messages(
                    model=llm.model,
                    hf_tokenizer=hf_tokenizer,
                    state_messages=state_thought_messages,
                    state_action_messages=state_thought_act_messages,
                    state_continue_final_message=True,
                )
                act_logprobs: list[list[float]] = _state_thought_act_outputs[0]
                # state_thought_act_toks: list[list[int]]   = _state_thought_act_outputs[1]
                act_probs_in_group = [np.exp(sum(_logps) / len(_logps)) for _logps in act_logprobs]
                rewards_in_group = self._compute_group_rewards(
                    rewards_in_group=act_probs_in_group,
                    split=split,
                )
                # Pick the highest-reward thought to continue for the next step
                best_thought_idx = np.argmax(rewards_in_group).item()
                best_thought = thoughts_in_group[best_thought_idx]
                model_messages = [{"role": "assistant", "content": best_thought}]
                parsed_actions: list[ActionFromLLM] = get_actions(model_messages)

                env_step_result: EnvironmentStepResult = env.step(
                    parsed_actions=parsed_actions,
                    current_state=state,
                    current_messages=state_messages,
                    reward=rewards_in_group[best_thought_idx],
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

                # rich_print_messages(hf_tokenizer.decode(batch_state_input_ids[0]))
                # breakpoint()

                # ---------- Save episode steps for each generation ----------
                shared_kwargs = {
                    "next_obs": next_obs,
                    "tools": state.tools,
                    "temperature": temperature,
                    "done": done,
                    "truncated": truncated,
                    "timestep": state.timestep,
                    "try_step": try_step,
                    "batch_id": batch_id,
                    "unique_data_sample_id": sample_id,
                    "split": split,
                    "action_probs_in_group": act_probs_in_group,
                    "discount_factor": discount_factor,
                }

                # 1. Save (state, thought, action) artifacts
                _state_messages_in_group = [standard_chat] * num_return_sequences
                _state_thought_act_outputs = get_act_logprobs_and_state_act_tokens_from_messages(
                    model=llm.model,
                    hf_tokenizer=hf_tokenizer,
                    state_messages=_state_messages_in_group,
                    state_action_messages=state_thought_act_messages,
                    state_continue_final_message=False,
                )
                thought_action_logprobs:  list[list[float]] = _state_thought_act_outputs[0]
                state_thought_act_tokens: list[list[int]]   = _state_thought_act_outputs[1]

                state_thought_act_trajectory_group = _get_trajectory_group_from_generations(
                    state_messages_in_group=_state_messages_in_group,
                    actions_in_group=[msg[0] for msg in thought_act_messages],  # list[dict[str, str]]
                    state_len_in_group=[state_len] * num_return_sequences,
                    state_action_tokens_in_group=state_thought_act_tokens,
                    old_logprobs_in_group=thought_action_logprobs,
                    rewards_in_group=rewards_in_group,
                    generation_ids_in_group=range(num_return_sequences),
                    get_trajectory_group_method=self._get_trajectory_group,
                    **shared_kwargs,
                )
                all_thought_act_trajectory_groups.append(state_thought_act_trajectory_group)

                # 2. Save action-prompted (state, action, thought) steps for RL'ing Act-PRM model
                state_action_thought_messages_G = [
                    deepcopy(act_prompt_state_messages) for _ in range(num_return_sequences)
                ]
                for i, _thought_text in enumerate(thoughts_in_group):
                    state_action_thought_messages_G[i][-1]["content"] += _thought_text
                
                _state_messages_in_group = [act_prompt_state_messages] * num_return_sequences
                _state_act_thought_outputs = get_act_logprobs_and_state_act_tokens_from_messages(
                    model=llm.model,
                    hf_tokenizer=hf_tokenizer,
                    state_messages=_state_messages_in_group,
                    state_action_messages=state_action_thought_messages_G,
                    state_continue_final_message=True,
                )
                thought_logprobs:         list[list[float]] = _state_act_thought_outputs[0]
                state_act_thought_tokens: list[list[int]]   = _state_act_thought_outputs[1]

                _actions_in_group = [
                    {"role": "assistant", "content": _thought_text} for _thought_text in thoughts_in_group
                ]
                state_act_thought_trajectory_group = _get_trajectory_group_from_generations(
                    state_messages_in_group=_state_messages_in_group,
                    actions_in_group=_actions_in_group,  # list[dict[str, str]], not exactly what we want, but we get thru state_action_tokens_in_group
                    state_len_in_group=[state_input_len_left_padded] * num_return_sequences,
                    state_action_tokens_in_group=state_act_thought_tokens,
                    old_logprobs_in_group=thought_logprobs,
                    rewards_in_group=rewards_in_group,
                    generation_ids_in_group=range(num_return_sequences),
                    get_trajectory_group_method=self._get_trajectory_group,
                    **shared_kwargs,
                )
                all_act_thought_trajectory_groups.append(state_act_thought_trajectory_group)

                # Transition to next state
                state = next_state

                # Visualize generated thoughts
                if self.verbose:
                    for i in [best_thought_idx]:  # or range(num_return_sequences), but this is a lot
                        header_text = (
                            f"Batch {batch_id}, Split {split}, Try {try_step}, "
                            f"Sample {sample_id}, Generation {i}, "
                            f"Step {state.timestep - 1} / {max_turns - 1}, "  # -1 bc took a step
                            f"Reward {rewards_in_group[i]:.4f}"
                        )
                        rewards_str = ", ".join([f"{r:.4f}" for r in sorted(rewards_in_group)[::-1]])
                        panel_content = [
                            f"Rewards: [bright_green][{rewards_str}][/bright_green]",
                            f"Run url: [cyan]{self.run_url}[/cyan]",
                            f"Run cmd: [bright_blue]{self.run_cmd}[/bright_blue]",
                        ]
                        if self.name_or_identifier:
                            panel_content.append(
                                f"Name/ID: [bright_yellow]{self.name_or_identifier}[/bright_yellow]"
                            )
                        if self.cfg.get("dataset_url_sft", None) is not None:
                            panel_content.append(
                                f"SFT url: [cyan]{self.cfg.dataset_url_sft}[/cyan]"
                            )
                        panel_content = "\n".join(panel_content)

                        state_thought_act_text = hf_tokenizer.decode(
                            state_thought_act_tokens[i],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        )
                        state_act_thought_text = hf_tokenizer.decode(
                            state_act_thought_tokens[i],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        )
                        try:
                            rich_print_messages(
                                msg_text=state_thought_act_text,
                                bos_token="<|im_start|>",  # hardcoded for Qwen models
                                eos_token="<|im_end|>\n",
                                assistant_color=ROYGBIV[i % len(ROYGBIV)],
                            )
                            console.print(Panel(panel_content, title=header_text, style="bold"))
                        except Exception as e:
                            logger.error(f"{e.__class__.__name__}: {e}")
                            print(state_thought_act_text)
                            print(f"{"-" * 50} {header_text} {"-" * 50}")
                            print(panel_content)
                        try:
                            rich_print_messages(
                                msg_text=state_act_thought_text,
                                bos_token="<|im_start|>",
                                eos_token="<|im_end|>\n",
                                assistant_color=ROYGBIV[i % len(ROYGBIV)],
                            )
                            console.print(Panel(panel_content, title=header_text, style="bold"))
                        except Exception as e:
                            logger.error(f"{e.__class__.__name__}: {e}")
                            print(state_act_thought_text)
                            print(f"{"-" * 50} {header_text} {"-" * 50}")
                            print(panel_content)
        
        if was_training:  # assume single try for now
            llm.model.train()
                
        hf_tokenizer.padding_side = og_tokenizer_padding_side     
        return {
            "policy": all_act_thought_trajectory_groups, 
            "think_act_policy": all_thought_act_trajectory_groups,
        }
    
