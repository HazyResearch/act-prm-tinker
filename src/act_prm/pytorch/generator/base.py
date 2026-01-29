"""
Base class for generation / rollout sampling for Hugging Face Transformers models
"""

import sys
from copy import copy, deepcopy
from typing import Any, Callable

from omegaconf import DictConfig
# from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

import torch
import torch.nn.functional as F
from tinker_cookbook.utils import ml_log
from transformers import PreTrainedTokenizerBase

from act_prm.environments import Environment, EnvironmentState, EnvironmentStepResult
from act_prm.llm_handlers import HuggingFaceLLM
from act_prm.llm_handlers.action_utils import get_actions
from act_prm.replay_buffer.types import (
    EpisodeStep, Trajectory, TrajectoryGroup, MeanCenteredTrajectoryGroup
)
from act_prm.utils.display import RichTextStreamer


console = Console()
ROYGBIV = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"]


def get_response_content(msg: dict[str, Any]) -> str:
    """
    Get message content from an Environment response message
    """
    return msg["output"] if msg.get("output", None) else msg["content"]


def get_action_logprobs_and_state_action_tokens(
    logits: torch.FloatTensor,
    input_ids: torch.LongTensor,
    attention_mask: torch.BoolTensor,  # should be, but may be LongTensor
    state_lens: list[int],
    **kwargs: Any,
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Get action logprobs from logits and input_ids (i.e., labels or target token ids)
    
    Assumes:
    - logits are right-padded and shape (batch_size, seq_len, vocab_size)
    - input_ids is shape (batch_size, seq_len)
    - state_lens is list of length batch_size, each element is len(state_tokens)

    Returns:
    - logprobs, which is a list of length batch_size, each element is a list of length
      individual_action_len (the number of action tokens in the generation)
    """
    logprobs = F.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size) -> (B, L, V)
    labels = input_ids  # convenience alias
    attention_mask = attention_mask.bool()
    # Tmask: 1111111111111111111111111111111111
    # state: we linear (state_len = 9)
    # total: we linearized the chungus among us 
    # input: we linearized the chungus among u `total[:-1]`
    # label: e linearized the chungus among us `total[1:]`
    # start: --------ized the chungus among us `label[Tmask[1:]][state_len - 1:]`
    # state: every (state_len = 5)
    # Tmask: 1111111111111111111111000000000000
    # total: everything is chungus.xxxxxxxxxxxx
    # input: everything is chungus.xxxxxxxxxxx `total[:-1]`
    # label: verything is chungus.xxxxxxxxxxxx `total[1:]`
    # Lmask: 111111111111111111111000000000000 `Tmask[1:]`
    # label: verything is chungus.             `label[Tmask[1:]]`
    # start: ----thing is chungus.             `label[Tmask[1:]][state_len - 1:]`
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    logprobs = logprobs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, L - 1)

    # Get logprobs for action tokens only
    a_starts = [state_len - 1 for state_len in state_lens]
    logprobs = [
        logprobs[b_idx][attention_mask[b_idx, 1:]][start_idx:].tolist()  # mask matches targets
        for b_idx, start_idx in enumerate(a_starts)
    ]
    # MZ 1/27/26: maybe more clear to keep separate?
    # logprobs = [logprobs[b_idx][attn_mask] for b_idx, attn_mask in enumerate(attention_mask)]
    # logprobs = [logprobs[b_idx][start_idx:] for b_idx, start_idx in enumerate(a_starts)]
    state_action_tokens = [
        # input_ids[b_idx][attention_mask[b_idx]][start_idx:].tolist()
        input_ids[b_idx][attention_mask[b_idx]].tolist()
        for b_idx, start_idx in enumerate(a_starts)
    ]  # inputs should be action_tokens[b_idx][:-1], targets action_tokens[b_idx][1:]
    return logprobs, state_action_tokens


def get_batch_model_inputs(
    input_messages: list[list[dict[str, str]]],
    tools: list[list[dict[str, str]]] | None,
    hf_tokenizer: PreTrainedTokenizerBase,
    padding_side: str = "left",
    enable_thinking: bool = False,
    is_train: bool = True,
) -> tuple[dict[str, Any], list[int], bool]:
    """
    Get model_inputs (input_ids, attention_mask, etc.) from a batch of input messages

    Returns:
    - batch_model_inputs: Transformer model inputs (input_ids, attention_mask, etc.)
    - input_lens: list of length batch_size, each element is number of non-padded input tokens
    - drop_last_batch_item: Whether to drop the last item of the batch
                            (for logprobs/inference; see below)
    """
    hf_tokenizer.padding_side = padding_side
    drop_last_batch_item = False
    assert len(input_messages) == len(tools), (
        "Each input message must have corresponding tool descriptions (can be list[None])"
    )
    
    # First, only format text to handle different available tools per state
    batch_model_texts = [
        hf_tokenizer.apply_chat_template(
            input_messages[b_idx],
            tools=tools[b_idx] if tools is not None else None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            padding=False,
            tokenize=False,
        )
        for b_idx in range(len(input_messages))
    ]
    if len(batch_model_texts) == 1 and is_train:
        # For logprobs/inference, HF Transformers can treat padding / batch_size >1
        # differently..., so we add dummy message to be consistent (always pad)
        batch_model_texts.append({"role": "assistant", "content": ""})
        drop_last_batch_item = True

    # Then tokenize (get padded input_ids, attention_mask, etc.)
    batch_model_inputs = hf_tokenizer(
        batch_model_texts,
        padding=True,
        return_tensors="pt"
    )
    input_lens = [attn_mask.sum().item() for attn_mask in batch_model_inputs["attention_mask"]]
    return batch_model_inputs, input_lens, drop_last_batch_item


def _get_trajectory_group_from_generations(
    state_messages_in_group: list[list[dict[str, str]]],
    actions_in_group: list[dict[str, str]],
    state_len_in_group: list[int],
    state_action_tokens_in_group: list[list[int]],
    old_logprobs_in_group: list[list[float]],
    rewards_in_group: list[float],
    action_probs_in_group: list[float],
    generation_ids_in_group: list[int],
    try_step: int,
    discount_factor: float,  # self.discount_factor
    get_trajectory_group_method: Callable,
    **shared_kwargs: Any,
) -> TrajectoryGroup:
    """
    Save generations to a TrajectoryGroup
    """
    # For some cases, state will be the same for all generations
    if len(state_messages_in_group) == 1:
        state_messages_in_group = [state_messages_in_group[0]] * len(actions_in_group)
        state_len_in_group = [state_len_in_group[0]] * len(actions_in_group)
    
    episode_steps_in_group: list[EpisodeStep] = [
        EpisodeStep(
            state=state_messages_in_group[_idx],
            action=action,  # dict[str, str]
            state_action_tokens=state_action_tokens_in_group[_idx],
            state_len=state_len_in_group[_idx],
            old_logprobs=old_logprobs_in_group[_idx],
            reward=rewards_in_group[_idx],
            action_prob=action_probs_in_group[_idx],
            generation_id=generation_ids_in_group[_idx],
            try_step=try_step,
            **shared_kwargs,
        ) for _idx, action in enumerate(actions_in_group)
    ]
    trajectories_in_group: list[Trajectory] = [
        Trajectory(
            episode_steps=[episode_step],
            try_step=try_step,
            discount_factor=discount_factor,
            final_reward=rewards_in_group[i],
        ) for i, episode_step in enumerate(episode_steps_in_group)
    ]
    # self._get_trajectory_group
    return get_trajectory_group_method(
        trajectories=trajectories_in_group,
        final_rewards=rewards_in_group,
        discount_factor=discount_factor,
    )


class HuggingFaceGenerator:
    """
    Compute rollouts using Hugging Face Transformers models
    """
    def __init__(
        self,
        llm: HuggingFaceLLM,
        hf_tokenizer: PreTrainedTokenizerBase,
        env: Environment,
        cfg: DictConfig,
        enable_thinking: bool | None = None,  # default to HF template, but often set to False
        discount_factor: float | None = None,
        mean_center: bool = False,
        ml_logger: ml_log.Logger | None = None,
        name_or_identifier: str | None = None,
        streamer: bool = False,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.hf_tokenizer = hf_tokenizer
        self.env = env
        self.cfg = cfg

        self.enable_thinking = enable_thinking
        self.run_url = ml_logger.get_logger_url() if ml_logger is not None else cfg.get("run_url", None)
        # self.run_cmd = f"uv run python main.py {" ".join(sys.argv[1:])}"
        self.run_cmd = cfg.get("run_cmd", " ".join(sys.argv))
        self.name_or_identifier = name_or_identifier
        self.discount_factor = discount_factor or cfg.get("discount_factor", 0.9)
        self.mean_center = mean_center  # mean-center the advantages

        # Silly streaming
        self.streamer = RichTextStreamer(
            self.hf_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        ) if streamer else None
        self.verbose = verbose

    def _get_messages_from_state(
        self,
        state: EnvironmentState,
        default_context: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get messages from the environment state, in the form of
        [{"role": <role>, "content": <content>}, ...]
        """
        # Initialize as prior observations + model's last response + environment new messages
        messages = (
            (state.prior_messages or []) 
            + (state.model_response or [])
            + state.new_messages
        )
        # Preprocess into {"role": <role>, "content": <content>} format
        # -> See `act_prm.environments` classes for environment responses
        messages = [
            {"role": msg["role"], "content": msg["output"]}
            if msg.get("type", "") == "function_call_output"
            else msg
            for msg in messages
        ]
        # Add default context (few-shot examples) if provided
        default_context = deepcopy(default_context or [])
        # Remove system prompt (will add it back after default context)
        if messages[0].get("role", "") == "system":
            messages = messages[1:]
        # Return final messages list
        return [
            {"role": "system", "content": state.system_prompt},
            *default_context,
            *messages,
        ]

    def _get_trajectory_group(self, **kwargs: Any) -> TrajectoryGroup:
        """
        Return trajectory group class
        """
        if self.mean_center:
            # Returns trajectory group where we compute advantages by:
            # 1. Computing mean-centered final rewards: final_reward - mean(final_rewards)
            # 2. Optionally apply step-wise discounting to these values
            return MeanCenteredTrajectoryGroup(**kwargs)
        return TrajectoryGroup(**kwargs)    

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
        # start_idx: int = 0,
        num_return_sequences: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        pbar_position: int = 0,
    ) -> dict[str, list[TrajectoryGroup]]:
        """ 
        Run rollouts for a single batch, e.g., by generating rollouts and grading them

        Returns:
        - new_trajectories: Trajectories for the batch, keyed by an identifier (default "policy")
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
        # Also keep track of original pad token id bc we'll change it for logprobs
        # og_pad_token_id = copy(hf_tokenizer.pad_token_id)
        # pad_token_id = -og_pad_token_id  # try this for logprobs matching

        # Generation parameters
        max_tokens = max_tokens or cfg.max_tokens
        temperature = temperature or cfg.temperature
        num_return_sequences = num_return_sequences or (
            cfg.group_size if split == "train" else cfg.eval_group_size
        )

        with torch.no_grad():
            # Generate rollouts
            # -> We generate until the last generation is done, keeping track of:
            #    1. All generation indices (1, ..., num_return_sequences) (fixed) (all_gen_ids)
            #    2. Batch of generation indices that are not done (gets smaller) (gen_ids_todo)
            all_episode_steps: list[list[EpisodeStep]] = [[] for _ in range(num_return_sequences)]
            all_final_rewards: list[float] = [0.0 for _ in range(num_return_sequences)]
            
            all_gen_ids = list(range(num_return_sequences))  # this is fixed, i.e., [1, 2, 3, ...]
            gen_ids_todo = list(range(num_return_sequences))  # this can get smaller
            # unique_data_sample_ids = [sample_id * num_return_sequences + gen_id for gen_id in all_gen_ids]
            num_todo = len(gen_ids_todo)

            batch_states: list[EnvironmentState] = [
                env.reset(sample_idx=sample_id, generation_idx=gen_idx, try_step=try_step)
                for _, gen_idx in enumerate(gen_ids_todo)
            ]

            pbar_task = tqdm(
                total=num_return_sequences,
                desc=f"Generating rollout 0 / {num_return_sequences - 1} ({num_todo} left)",
                colour="cyan",
                leave=False,
                position=pbar_position,
            )
            rollout_pbars = [
                tqdm(
                    total=env.max_turns,
                    desc=f"Generating rollout {gen_id}",
                    colour=ROYGBIV[gen_id % len(ROYGBIV)],
                    leave=True,
                    position=pbar_position + gen_id + 1,
                )
                for gen_id in gen_ids_todo
            ]
            # while not all(batch_dones):
            while len(gen_ids_todo) > 0:
                batch_state_messages = [
                    self._get_messages_from_state(state=state, default_context=None)
                    for state in batch_states
                ]
                batch_state_inputs, state_input_lens, _ = get_batch_model_inputs(
                    input_messages=batch_state_messages,
                    tools=[state.tools for state in batch_states],
                    hf_tokenizer=hf_tokenizer,
                    padding_side="left",
                    enable_thinking=self.enable_thinking,
                    is_train=split == "train",
                )
                state_input_len_left_padded = batch_state_inputs["input_ids"].shape[1]

                # Generate model responses
                # (batch_size, max_input_len) -> (batch_size, max_input_len + max_new_tokens)
                outputs = llm.model.generate(
                    **batch_state_inputs.to(device),
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=hf_tokenizer.pad_token_id,
                    # output_scores=True,  # returns logprobs, but we'll recompute for training match
                    # output_logprobs=True,  # MZ: I can't tell if above supported though, so just use logprobs
                    # Silly streaming only supports batch_size == 1
                    streamer=self.streamer if len(state_input_lens) == 1 else None,
                )
                # Decode and convert tokens to messages
                decoded_texts = hf_tokenizer.batch_decode(
                    outputs[:, state_input_len_left_padded:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                # For now we only allow one tool call per response
                decoded_texts = [
                    f"{text.split(llm.tool_call_eos)[0]}{llm.tool_call_eos}"
                    if llm.tool_call_eos in text
                    else text
                    for text in decoded_texts
                ]
                batch_model_messages: list[list[dict[str, str]]] = [
                    [{"role": "assistant", "content": text}] for text in decoded_texts
                ]
                batch_state_action_messages = [
                    batch_state_messages[_idx] + batch_model_messages[_idx]
                    for _idx in range(len(batch_state_messages))
                ]
                # Get logprobs for action tokens
                # -> 1. Get token_ids for all state_action generations
                batch_state_action_inputs, _, drop_last_batch_item = get_batch_model_inputs(
                    input_messages=batch_state_action_messages,
                    tools=[state.tools for state in batch_states],
                    hf_tokenizer=hf_tokenizer,
                    padding_side="right",
                    enable_thinking=self.enable_thinking,
                    is_train=split == "train",
                )
                # -> 2. Compute model inference logprobs (matches those at training time)
                logits = llm.model(**batch_state_action_inputs.to(device), use_cache=False).logits
                if drop_last_batch_item:  # now we remove last item after model inference
                    logits = logits[:-1]
                    state_input_lens = state_input_lens[:-1]
                    for k, v in batch_state_action_inputs.items():
                        setattr(batch_state_action_inputs, k, v[:-1])
                
                act_logps_and_state_act_toks = get_action_logprobs_and_state_action_tokens(
                    logits=logits,
                    state_lens=state_input_lens,
                    **batch_state_action_inputs.to(device),  # input_ids, attention_mask
                )
                act_logprobs: list[list[float]] = act_logps_and_state_act_toks[0]       # batch_size x action_len
                state_action_tokens: list[list[int]] = act_logps_and_state_act_toks[1]  # batch_size x (state_len + action_len)

                # Transition to next states
                # -> Parse to consistent ActionFromLLM format
                batch_parsed_actions = [get_actions(msgs) for msgs in batch_model_messages]
                batch_env_step_results: list[EnvironmentStepResult] = [
                    env.step(
                        parsed_actions=batch_parsed_actions[_idx],
                        model_response=batch_model_messages[_idx],
                        current_state=state,
                        current_messages=batch_state_messages[_idx],
                    )
                    for _idx, state in enumerate(batch_states)
                ]
                batch_next_states = [_result.state for _result in batch_env_step_results]
                batch_rewards = [_result.reward for _result in batch_env_step_results]
                batch_next_obs = [
                    [
                        {"role": msg["role"], "content": get_response_content(msg)}
                        for msg in next_state.new_messages
                    ]
                    for next_state in batch_next_states
                ]
                if self.verbose:
                    max_to_display = 1  # hardcoded hack
                    for _idx in range(len(batch_next_obs))[:max_to_display]:
                        color = f"italic color({_idx % 8 + 8})"
                        _header = f"Next Observation {_idx} (Timestep {batch_states[_idx].timestep})"
                        print("\n")
                        console.print(Panel(batch_next_obs[0][0]["content"], title=_header, style=color))
                # MZ 1/27/26 NOTE: this is a bit heinous...
                batch_episode_steps = [
                    EpisodeStep(
                        state=batch_state_messages[_idx],
                        action=batch_model_messages[_idx][0],  # dict[str, str]
                        next_obs=batch_next_obs[_idx],
                        tools=batch_states[_idx].tools,
                        state_action_tokens=state_action_tokens[_idx],
                        state_len=state_input_lens[_idx],
                        old_logprobs=act_logprobs[_idx],
                        temperature=temperature,
                        reward=batch_env_step_results[_idx].reward,
                        done=batch_env_step_results[_idx].done,
                        truncated=batch_env_step_results[_idx].truncated,
                        timestep=batch_states[_idx].timestep,
                        try_step=batch_states[_idx].try_step,
                        batch_id=batch_id,
                        # unique_data_sample_id=unique_data_sample_ids[_idx],
                        unique_data_sample_id=sample_id,
                        generation_id=gen_id,
                        split=split,
                    )
                    for _idx, gen_id in enumerate(gen_ids_todo)
                ]
                # Build sequence of EpisodeSteps for each generation
                for _idx, gen_id in enumerate(gen_ids_todo):
                    all_episode_steps[gen_id].append(batch_episode_steps[_idx])
                    all_final_rewards[gen_id] = batch_rewards[_idx]  # Overwrite til last reward

                # Update pbars and finished rollouts
                for _idx, gen_id in enumerate(gen_ids_todo):
                    rollout_pbars[_idx].update(1)
                    if batch_env_step_results[_idx].done:
                        rollout_pbars[_idx].close()
                        rollout_pbars.pop(_idx)
                        gen_ids_todo.pop(_idx)
                # # Update gen_ids_todo to only include non-done generations
                # gen_ids_todo = [
                #     gen_id
                #     for _idx, gen_id in enumerate(gen_ids_todo)
                #     if not batch_env_step_results[_idx].done
                # ]
                # Move to next state
                batch_states = [
                    next_state
                    for _idx, next_state in enumerate(batch_next_states)
                    if not batch_env_step_results[_idx].done
                ]
                num_done = num_todo - len(gen_ids_todo)
                num_todo = len(gen_ids_todo)
                pbar_task.update(num_done)
                pbar_task.set_description(
                    f"Generating rollout {num_done} / {num_return_sequences - 1} ({num_todo} left)"
                )
                

            trajectories_in_group = [
                Trajectory(
                    episode_steps=all_episode_steps[gen_id],
                    final_reward=all_final_rewards[gen_id],
                    try_step=try_step,
                    discount_factor=discount_factor,
                ) for gen_id in range(num_return_sequences)
            ]
            all_trajectory_groups = [
                self._get_trajectory_group(
                    trajectories=trajectories_in_group,
                    final_rewards=all_final_rewards,
                    discount_factor=discount_factor,
                )
            ]
        if was_training:  # assume single try for now
            llm.model.train()
                
        hf_tokenizer.padding_side = og_tokenizer_padding_side     
        return {"policy": all_trajectory_groups}
    