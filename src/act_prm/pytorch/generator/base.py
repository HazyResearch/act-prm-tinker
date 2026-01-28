"""
Base class for generation / rollout sampling for Hugging Face Transformers models
"""

import sys
from copy import copy, deepcopy
from typing import Any, Callable

import torch
import torch.nn.functional as F

from omegaconf import DictConfig
from tqdm import tqdm

from tinker_cookbook.utils import ml_log
from transformers import PreTrainedTokenizerBase

from act_prm.environments import Environment, EnvironmentState, EnvironmentStepResult
from act_prm.llm_handlers import HuggingFaceLLM
from act_prm.llm_handlers.action_utils import get_actions
from act_prm.replay_buffer.types import (
    EpisodeStep, Trajectory, TrajectoryGroup, MeanCenteredTrajectoryGroup
)


def get_response_content(msg: dict[str, Any]) -> str:
    """
    Get message content from an Environment response message
    """
    return msg["output"] if msg.get("output", None) else msg["content"]


def get_action_logprobs(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    state_lens: list[int],
    pad_token_id: int,
) -> list[list[float]]:
    """
    Get action logprobs from logits and labels (target token ids)
    
    Assumes:
    - logits are left-padded is shape (batch_size, seq_len, vocab_size)
    - labels is shape (batch_size, seq_len)
    - state_lens is list of length batch_size, each element is len(state_tokens)
    - pad_token_id is the padding token id
      -> should be distinct from anything in labels, e.g., -pad_token_id

    Returns:
    - logprobs, which is a list of length batch_size, each element is a list of length
      individual_action_len (the number of action tokens in the generation)
    """
    logprobs = F.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size) -> (B, L, V)
    # we linearized the chungus among u
    # e linearized the chungus among us
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    logprobs = logprobs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, L - 1)

    # Get logprobs for action tokens only
    action_starts = [state_len - 1 for state_len in state_lens]
    logprobs = [logprobs[start_ix:] for start_ix in action_starts]
    act_mask = [labels[start_ix:] != pad_token_id for start_ix in action_starts]
    logprobs = [logprobs[b_idx][act_mask[b_idx]].tolist() for b_idx in range(len(logprobs))]
    return logprobs


def get_batch_model_inputs(
    input_messages: list[list[dict[str, str]]],
    tools: list[list[dict[str, str]] | None],
    hf_tokenizer: PreTrainedTokenizerBase,
    padding_side: str = "left",
) -> tuple[dict[str, Any], bool]:
    """
    Get model_inputs (input_ids, attention_mask, etc.) from a batch of input messages

    Returns:
    - batch_model_inputs: dict[str, Any]
    - drop_last_batch_item: bool
      -> If True, we need to drop the last item of the batch for logprobs/inference
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
            tools=tools[b_idx],
            add_generation_prompt=True,
            enable_thinking=False,
            padding=False,
            tokenize=False,
        )
        for b_idx in range(len(input_messages))
    ]
    if len(batch_model_texts) == 1:
        # For logprobs/inference, HF Transformers can treat padding / batch_size >1
        # differently..., so we add dummy message to be consistent (always pad)
        batch_model_texts.append({"role": "assistant", "content": ""})
        drop_last_batch_item = True

    # Then tokenize (get padded input_ids, attention_mask, etc.)
    batch_model_inputs = hf_tokenizer(
        batch_model_texts,
        padding=True,
        return_dict=True,
        return_tensors="pt"
    )
    return batch_model_inputs, drop_last_batch_item


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
        ml_logger: ml_log.Logger | None = None,
        name_or_identifier: str | None = None,
    ) -> None:
        self.llm = llm
        self.hf_tokenizer = hf_tokenizer
        self.env = env
        self.cfg = cfg

        self.enable_thinking = enable_thinking
        self.run_url = ml_logger.get_logger_url() if ml_logger is not None else None
        # self.run_cmd = f"uv run python main.py {" ".join(sys.argv[1:])}"
        self.run_cmd = " ".join(sys.argv)
        self.name_or_identifier = name_or_identifier

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
        batch_id: int,
        llm: HuggingFaceLLM | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        env: Environment | None = None,
        cfg: DictConfig | None = None,
        split: str = "train",
        num_tries: int = 1,
        start_idx: int = 0, 
        # tasks_per_update: int | None = None,
        name_or_identifier: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        pbar_position: int = 0,
    ) -> tuple[dict[str, Any], dict[str, list[Trajectory]]]:
        """ 
        Run rollouts for a single batch, e.g., by generating rollouts and grading them

        Returns:
        - final_metrics: Metrics for the batch, keyed by "{split}/{try_idx}/{metric}"
        - new_trajectories: Trajectories for the batch, keyed by an identifier (default "policy")
        """
        llm = llm or self.llm
        env = env or self.env
        cfg = cfg or self.cfg

        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        # Keep track of original padding side bc we'll change it multiple times below
        og_tokenizer_padding_side = hf_tokenizer.padding_side
        hf_tokenizer.padding_side = "left"  # confirm left-padding for generation
        pad_token_id = -hf_tokenizer.pad_token_id  # try this for logprobs matching
        
        env.split = split  # Select task split
        was_training = llm.model.training
        llm.model.eval()

        # Generation parameters
        max_tokens = max_tokens or cfg.max_tokens
        temperature = temperature or cfg.temperature
        num_return_sequences = cfg.group_size if split == "train" else cfg.eval_group_size
        
        # batch_size = tasks_per_update or len(env)  # len(env) is the number of tasks or problems
        all_eval_metrics = {}
        keys_for_correct = []
        eval_metric_keys = [
            "final_reward", "first_return", "action_prob", "last_state_len",
            "timesteps", "correct", "total",
        ]
        # Store new trajectories to return
        new_trajectories: dict[str, list[Trajectory]] = {}

        with torch.no_grad():
            for try_idx in range(num_tries):  # assume this is 1 for now
                # Generate rollouts
                # -> We generate until the last generation is done, keeping track of:
                #    1. All generation indices (1, ..., num_return_sequences) (fixed) (generation_ids)
                #    2. Batch of generation indices that are not done (gets smaller) (batch_ids_todo)

                all_episode_steps: list[list[EpisodeStep]] = [[] for _ in range(num_return_sequences)]
                all_final_rewards: list[float] = [0.0 for _ in range(num_return_sequences)]
                
                generation_ids = list(range(num_return_sequences))  # this is fixed, i.e., [1, 2, 3, ...]
                batch_ids_todo = list(range(num_return_sequences))  # this can get smaller
                unique_data_sample_ids = [start_idx + _id for _id in generation_ids]
                num_todo = len(batch_ids_todo)

                batch_states: list[EnvironmentState] = [
                    env.reset(sample_idx=start_idx, generation_idx=gen_idx, try_step=try_idx)
                    for _, gen_idx in enumerate(batch_ids_todo)
                ]

                pbar_task = tqdm(
                    total=num_return_sequences,
                    desc=f"Generating rollouts (try {try_idx})",
                    colour="yellow",
                    leave=False,
                    position=pbar_position,
                )
                # while not all(batch_dones):
                while len(batch_ids_todo) > 0:
                    batch_state_messages = [
                        self._get_messages_from_state(state=state, default_context=None)
                        for state in batch_states
                    ]
                    batch_state_inputs, _ = get_batch_model_inputs(
                        input_messages=batch_state_messages,
                        tools=[state.tools for state in batch_states],
                        hf_tokenizer=hf_tokenizer,
                        padding_side="left",
                    )
                    state_input_len = batch_state_inputs["input_ids"].shape[1]

                    # Generate model responses
                    outputs = llm.model.generate(
                        **batch_state_inputs.to(llm.model.device),
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        num_return_sequences=1,
                        pad_token_id=pad_token_id,
                        # output_scores=True,  # returns logprobs, but we'll recompute for training match
                        output_logprobs=True,  # MZ I can't tell if above supported though, so just use logprobs
                    )
                    # Decode and convert tokens to messages
                    decoded_texts = hf_tokenizer.batch_decode(
                        outputs[:, state_input_len:],
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
                    batch_model_messages = [
                        [{"role": "assistant", "content": text}] for text in decoded_texts
                    ]
                    
                    # Get logprobs
                    # First get token_ids for all state_action generations
                    batch_state_action_inputs, drop_last_batch_item = get_batch_model_inputs(
                        input_messages=batch_state_messages + batch_model_messages,
                        tools=[state.tools for state in batch_states],
                        hf_tokenizer=hf_tokenizer,
                        padding_side="right",
                    )
                    
                    if split != "train":  # just get from the generation
                        # logprobs = F.log_softmax(outputs.logits, dim=-1)  # batch_size, seq_len, vocab_size
                        logits = outputs.logits
                        labels = batch_state_inputs["input_ids"]  # batch_size, seq_len

                        logprobs = get_action_logprobs(
                            outputs.logits,
                            outputs
                            labels, input_len, pad_token_id)
                        get_action_logprobs(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    state_lens: list[int],
    pad_token_id: int,
                        
                        
                        logprobs = -F.cross_entropy(outputs.logits, labels, reduction="none")
                        # Get action-only logprobs
                        logprobs = logprobs[:, input_len:, :]  # batch_size, padded_action_len, 1
                        act_pads = labels[:, input_len:] == pad_token_id  # still need first eos token tho...
                        logprobs = [logprobs[_idx][act_pads[_idx]] for _idx in range(len(logprobs))]
                        batch_model_logprobs = logprobs

                    else:  # Compute logprobs to match those at training / model inference
                    
                    
                    if split == "train":  # otherwise we won't store the logprobs
                        with torch.no_grad():
                            model_was_training = copy(llm.model.training)
                            llm.model.eval()
                            # To compute logprobs / during model inference,
                            # we want to align positional embeddings
                            og_tokenizer_padding_side = copy(hf_tokenizer.padding_side)
                            hf_tokenizer.padding_side = "right"

                            batch_state_inputs = hf_tokenizer(
                                batch_state_texts,
                                padding=True,
                                return_dict=True,
                            )

                        



                        if model_was_training:
                            llm.model.train()
                    




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
                    batch_next_states = [
                        env_step_result.state for env_step_result in batch_env_step_results
                    ]
                    batch_rewards = [
                        env_step_result.reward for env_step_result in batch_env_step_results
                    ]

                    batch_next_obs = []                
                    for state in batch_next_states:
                        batch_next_obs.append(
                            [
                                {"role": msg["role"], "content": get_response_content(msg)}
                                for msg in state.new_messages
                            ]
                        )
                    batch_episode_steps = [
                        EpisodeStep(
                            state=batch_state_messages[_idx],
                            action=batch_model_messages[_idx],
                            next_obs=batch_env_step_results[_idx].state.new_messages,
                            tools=batch_states[_idx].tools,
                            state_action_tokens=batch_state_inputs["input_ids"][_idx],
                            state_len=len(batch_state_inputs["input_ids"][_idx]),
                            # old_logprobs=batch_env_step_results[gen_idx].state.model_response.logprobs,
                            old_logprobs=None,  # TODO Mz 1/23/26 get theses
                            temperature=temperature,
                            reward=batch_env_step_results[_idx].reward,
                            done=batch_env_step_results[_idx].done,
                            truncated=batch_env_step_results[_idx].truncated,
                            timestep=batch_states[_idx].timestep,
                            try_step=batch_states[_idx].try_step,
                            batch_id=batch_id,
                            unique_data_sample_id=unique_data_sample_ids[_idx],
                            generation_id=gen_id,
                            split=split,
                        )
                        for _idx, gen_id in enumerate(batch_indices)
                    ]
                    # Build sequence of EpisodeSteps for each generation
                    for _idx, gen_id in enumerate(batch_indices):
                        all_episode_steps[gen_id].append(batch_episode_steps[_idx])
                        all_final_rewards[gen_id] = batch_rewards[_idx]  # will keep on overwriting til last reard

                    batch_indices = [
                        gen_id
                        for _idx, gen_id in enumerate(batch_indices)
                        if not batch_env_step_results[_idx].done
                    ]
                    # Move to next state
                    batch_states = [
                        batch_next_states[_idx]
                        for _idx, _ in enumerate(batch_indices)
                    ]
                    num_done = num_todo - len(batch_indices)
                    num_todo = len(batch_indices)
                    pbar_task.update(num_done)

                    

                
                trajectories_in_group = [
                    Trajectory(
                        episode_steps=all_episode_steps[gen_id],
                        final_reward=all_final_rewards[gen_id],
                        try_step=try_idx,
                        discount_factor=cfg.discount_factor,
                    ) for gen_id in range(num_return_sequences)
                ]
                all_trajectory_groups = [
                    # self._get_trajectory_group(
                    TrajectoryGroup(
                        trajectories=trajectories_in_group, 
                        # final_rewards=final_rewards_in_group,
                        discount_factor=cfg.discount_factor,
                    )
                ]
                if was_training:  # assume single try for now
                    llm.model.train()

                break # only one try for now
                
        hf_tokenizer.padding_side = og_tokenizer_padding_side
                
        return {"policy": all_trajectory_groups}
    