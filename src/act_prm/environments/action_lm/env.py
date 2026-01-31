"""
Supervised Fine-Tuning "Environment"

For now, we just use a way to load Hugging Face datasets and preprocess them for SFT 
"""

import logging
from copy import copy
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset, DatasetDict, load_dataset
from rich import print as rich_print
from transformers import PreTrainedModel  # , TextStreamer
from tqdm import tqdm

# from ...llm_handlers import ActionFromLLM

from act_prm.environments.act_prm.utils import get_thought_and_actions
from act_prm.environments.base import Environment
from act_prm.utils.display import RichTextStreamer
# from ..types import EnvironmentState, EnvironmentStepResult
from .utils import check_model_inputs, ROYGBIV


logger = logging.getLogger(__name__)


class ActionLmState:
    """
    State for Action Language Modeling tasks (Pydantic object)
    """
    def __init__(self, trajectory: HFDataset, trajectory_messages: HFDataset) -> None:
        self.trajectory = trajectory
        self.trajectory_messages = trajectory_messages

# class ActionLmStepResult(EnvironmentStepResult):
#     """
#     Step result for HotpotQA multiple choice tasks
#     """
#     state: ActionLmState
#     reward: float
#     done: bool
#     truncated: bool
#     info: dict[str, Any] | None = None

def _get_item(x: Any) -> Any:
    """
    Get item from a tensor or scalar
    """
    return x.item() if hasattr(x, "item") else x


class ActionLmEnv(Environment):
    """
    Action Language Modeling "Environment"
    
    From a static Hugging Face dataset, 
    """
    
    def __init__(
        self,
        dataset_config: dict[str, Any],
        actions_only: bool = False,
        build_full_states: bool = False,  # if True, do an extra step of filling in past observations for samples
        best_actions_only: bool = False,
        best_halves_only: bool = False,
        success_only: bool = True,
        max_timestep: int | None = None,
        target_thoughts_eval: bool = False,
        action_bos: str = "<tool_call>",
        action_eos: str = "</tool_call>",
        final_answer_bos: str = "Final Answer: ",
        frac_train_tasks: int = 0.8,
        frac_eval_tasks: int = 0.2,
        fp32_loss: bool = False,
        max_input_id_len: int | None = None,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = "You are a helpful assistant.",
        pretrained_model_config: dict[str, Any] | None = None,
        sample_id_name: str = "unique_data_sample_id",
        timestep_name: str = "timestep",
        generation_id_name: str = "generation_id",
        num_eval_gen_samples: int | None = None,
        num_eval_rollout_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_config = dataset_config
        self.actions_only = actions_only
        self.build_full_states = build_full_states
        self.best_actions_only = best_actions_only
        self.best_halves_only = best_halves_only
        self.success_only = success_only

        self.max_timestep = max_timestep
        self.max_input_id_len = max_input_id_len

        self.target_thoughts_eval = target_thoughts_eval  # if True, we compute inference metrics over thought tokens too

        self.sample_id_name = sample_id_name
        self.timestep_name = timestep_name
        self.generation_id_name = generation_id_name

        self.num_eval_gen_samples = num_eval_gen_samples
        self.num_eval_rollout_samples = num_eval_rollout_samples
        
        # Parse thoughts and actions from sample dataset
        self.action_bos = action_bos
        self.action_eos = action_eos
        self.final_answer_bos = final_answer_bos
        self.parse_action_kwargs = {
            "action_bos": self.action_bos,
            "action_eos": self.action_eos,
            "final_answer_bos": self.final_answer_bos,
        }
        self.system_prompt = system_prompt
        
        # Build environment
        self.frac_train_tasks = frac_train_tasks
        self.frac_eval_tasks = frac_eval_tasks
        self.seed = seed
        self.split = split

        # For loss computation
        self.fp32_loss = fp32_loss

        # Get tokenizer
        self.pretrained_model_config = pretrained_model_config
        self.tokenizer = self._init_tokenizer()  # see act_prm/environments/base.py
        
        # Load data
        self.system_prompt = system_prompt
        # We'll use the DataFrames for trajectory-specific evaluation
        self.datasets, self.df_train, self.df_eval = self.init_data()
        self.datasets_rl: dict[str, tuple[list[HFDataset], list[HFDataset]]] = {
            # "train": self.init_rl_data(self.df_train, split="train"),
            "eval":  self.init_rl_data(self.df_eval,  split="eval"),
        }

        # Silly streaming
        self.streamer = RichTextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        self.run_url: str | None = None
        self.run_cmd: str | None = None
        
    def __len__(self) -> int:
        """
        Load size of RL dataset split
        """
        # return len(self.datasets[self.split])
        # self.datasets_rl[self.split] is a tuple of (training data tasks, messages tasks) 
        return len(self.datasets_rl[self.split][1])

    def _add_group_metrics(self, df: pd.DataFrame, metric: str = "advantage") -> pd.DataFrame:
        """
        Add group metrics to each sample in a dataframe (assumed to be part of a sampling group)
        """
        m = df[metric]
        df["best_action"] = m == m.max()
        df[f"group_{metric}"] = m - m.mean()
        df["best_half"] = m > m.median()
        return df

    def _maybe_build_full_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build full states for each sample

        Assumes df has columns: [state, action, self.sample_id_name, self.timestep_name, self.generation_id_name]
        """
        # if not self.build_full_states:
        #     return df

        unique_sample_ids = df[self.sample_id_name].unique()
        dfs_by_sample_update = []  # keep track of updated dfs for each sample here (heinous copy)

        if self.generation_id_name not in df.columns:
            # MZ 1/25/26: hack, but if we don't have generation ids assume it's all 0
            df[self.generation_id_name] = 0
        
        for sample_id in unique_sample_ids:
            df_by_sample = df[df[self.sample_id_name] == sample_id]
            unique_timesteps      = df_by_sample[self.timestep_name].unique()
            unique_generation_ids = df_by_sample[self.generation_id_name].unique()
            unique_timesteps.sort()
            unique_generation_ids.sort()

            # Keep track + build full state of chat messages for each generation
            all_full_states = [[] for _ in unique_generation_ids]
            dfs_by_sample_step_update = []  # a bit heinous, but handle updates as copies
            for t in unique_timesteps:
                df_by_sample_step = (
                    df_by_sample[df_by_sample[self.timestep_name] == t].reset_index(drop=True)
                )
                # Add group metrics (i.e., indicator for best_action, best_half, group_advantage)
                df_by_sample_step = self._add_group_metrics(df_by_sample_step)
                for gen_idx, state in df_by_sample_step["state"].items():
                    last_obs: dict[str, str] = state[-1]  # should be the same across all states
                    all_full_states[gen_idx].append(last_obs)  # add latest obs, action
                    all_full_states[gen_idx].append(df_by_sample_step["action"][gen_idx])
                # Update the state for all samples in df_by_sample_step (omit last action)
                # df_by_sample_step["state_full_obs"] = [_state[:-1] for _state in all_full_states]
                if self.build_full_states:  # Update the original state to include full observations
                    # df_by_sample_step["state"] = [_state[:-1] for _state in all_full_states]
                    try:
                        df_by_sample_step["state"] = [self.maybe_hide_observations(_state[:-1]) for _state in all_full_states]
                    except Exception as e:
                        print(f"{e.__class__.__name__}: {e}")
                        breakpoint()
                # Otherwise, will should also do something where we just hide the prior observations
                # if t > 0:
                #     for _idx, msg in enumerate(df_by_sample_step["state"][0]): print(_idx, msg)
                #     breakpoint()
                # if t > 0:
                #     df_by_sample_step["state"] = [
                #         [
                #             self.maybe_hide_observations(_state[:-1], first_obs_to_show=2, last_obs_to_show=1)
                #             # msg for _idx, msg in enumerate(_state[:-1])
                #             # if msg["role"] in ["system", "assistant"]
                #             # else {"role": msg["role"], "content": msg["content"]} if _idx <= 1
                #         ]
                #         for _state in all_full_states
                #     ]
                #     breakpoint()
                dfs_by_sample_step_update.append(df_by_sample_step)

            df_by_sample = pd.concat(dfs_by_sample_step_update)  # reassign to updated version
            dfs_by_sample_update.append(df_by_sample)

        return pd.concat(dfs_by_sample_update).reset_index().drop(columns=["index"])

    def init_data(self) -> tuple[DatasetDict, pd.DataFrame, pd.DataFrame]:
        """
        Initialize dataset (from pre-downloaded file)
        Returns:
        - datasets: DatasetDict of questions and answers by (train, val, test) splits
        - df_train: DataFrame of train samples
        - df_eval:  DataFrame of eval samples
        """
        ds = load_dataset(**self.dataset_config)

        filter_actions = self.best_actions_only or self.best_halves_only

        # We want to process this datasets into sequences of trajectories, also their timesteps,
        # -> So that we can then split them into train and eval splits based on the tasks
        df = ds.to_pandas()  # easier to work with
        df = df[df["return_"] > 0] if self.success_only else df
        df = (
            df[df[self.generation_id_name] == 0] 
            if self.generation_id_name in df.columns and not filter_actions 
            else df
        )
        df = self._maybe_build_full_states(df)

        # Filter for samples corresponding to best actions or best-half of actions
        if self.best_actions_only:
            df = df[df["best_action"]]
        elif self.best_halves_only:
            df = df[df["best_half"]]
        
        # Filter for samples corresponding to max timestep
        if self.max_timestep is not None:
            df = df[df[self.timestep_name] < self.max_timestep]

        # Separate into train and eval splits (no test for SFT evals)
        unique_sample_ids = df[self.sample_id_name].unique()
        np.random.seed(self.seed)
        np.random.shuffle(unique_sample_ids)
        num_train = int(len(unique_sample_ids) * self.frac_train_tasks)
        train_sample_ids = unique_sample_ids[:num_train]
        eval_sample_ids  = unique_sample_ids[num_train:]

        # Load from these for sampling based evaluation
        df_train = df[df[self.sample_id_name].isin(train_sample_ids)]
        df_eval  = df[df[self.sample_id_name].isin(eval_sample_ids)]

        # Convert to Hugging Face datasets and apply tokenization preprocessing
        ds_train = HFDataset.from_pandas(df_train)
        ds_eval = HFDataset.from_pandas(df_eval)
        # Applies tokenization for training and model inference evaluation
        ds_train = ds_train.map(
            partial(self._preprocess_sample, target_thoughts=True, split="train"),   # labels include thoughts
            remove_columns=ds_train.column_names,
            # load_from_cache_file=False,
            load_from_cache_file=True,
            desc="Tokenizing SFT train split",
        )
        ds_eval = ds_eval.map(
            # partial(self._preprocess_sample, target_thoughts=False),  # labels include actions only
            partial(self._preprocess_sample, target_thoughts=self.target_thoughts_eval, split="eval"),
            remove_columns=ds_eval.column_names,
            # load_from_cache_file=False,
            load_from_cache_file=True,
            desc="Tokenizing SFT eval split",
        )
        # Filter out samples too long
        if self.max_input_id_len is not None:
            max_input_id_len = max(len(x["input_ids"]) for x in ds_train)
            print(f"Max input id length: {max_input_id_len}")
            ds_train = ds_train.filter(lambda x: len(x["input_ids"]) <= self.max_input_id_len)
            ds_eval  = ds_eval.filter(lambda x: len(x["input_ids"]) <= self.max_input_id_len)
            print(f"Filtered to {len(ds_train)} train samples and {len(ds_eval)} eval samples")
            # print(f"Train samples length: {len(ds_train[0]['input_ids'])}")
            # print(f"Eval samples length: {len(ds_eval[0]['input_ids'])}")
            # breakpoint()

        datasets = DatasetDict({
            "train": ds_train.with_format("torch"),
            "eval":  ds_eval.with_format("torch"),
            "test":  ds_eval.with_format("torch"),  # won't use, but same as eval split
        })

        return datasets, df_train, df_eval

    # def init_rl_data(self, df: pd.DataFrame, split: str) -> list[list[dict[str, Any]]]:
    def init_rl_data(self, df: pd.DataFrame, split: str) -> list[HFDataset]:
        """
        Initialize dataset for RL
        
        We organize the DataFrame into a list of "trajectory" Datasets, i.e., a list of steps
        a Dataset of samples corresponding to a single task rollout.
        
        Each sample in the trajectory Dataset is a model inference input
        following `self._preprocess_sample`

        We evaluate by:
        1. Sampling a single trajectory list (high-level "sample")
        2. For each model inference input, computing the perplexity, likelihood, and step-wise accuracy

        """
        # samples: list[list[dict[str, Any]]] = []  # (num_samples, num_timesteps, model input dict)
        samples: list[HFDataset] = []
        samples_messages: list[HFDataset] = []
        
        unique_sample_ids = df[self.sample_id_name].unique()
        target_thoughts = True if split == "train" or self.target_thoughts_eval else False

        def get_action_from_msg(msg: dict[str, str]) -> str:
            return msg["content"].split(self.action_bos)[-1].split(self.action_eos)[0].strip()

        # Get number of samples to evaluate
        num_eval_gen_samples = self.num_eval_gen_samples or len(unique_sample_ids)
        num_eval_rollout_samples = self.num_eval_rollout_samples or len(unique_sample_ids)
        num_rl_samples = min(num_eval_gen_samples, num_eval_rollout_samples)
        
        pbar = tqdm(
            unique_sample_ids[:num_rl_samples],
            desc=f"Initializing RL dataset for {split.title()} split",
            leave=True,
            colour="magenta",
            position=1,
        )
        for _sample_id in pbar:
            df_by_sample = df[df[self.sample_id_name] == _sample_id]
            df_by_sample = df_by_sample[df_by_sample["best_action"]]  # only keep one trajectory
            df_by_sample.sort_values(by=self.timestep_name, ascending=True, inplace=True)
            # df_by_sample = df_by_sample.reset_index(drop=True)
            ds_sample = HFDataset.from_pandas(df_by_sample)
            ds_sample = ds_sample.map(
                partial(self._preprocess_sample, target_thoughts=target_thoughts, split=split),
                remove_columns=ds_sample.column_names,
                # load_from_cache_file=False,
                load_from_cache_file=True,
                desc=f"Tokenizing RL Sample {_sample_id}",
            )
            samples.append(ds_sample.with_format("torch"))
            # Get message-based samples for generation-based evaluation
            df_by_sample["target_action"] = df_by_sample["action"].apply(get_action_from_msg)
            df_by_sample_messages = df_by_sample[["state", "action", "target_action", "tools"]].reset_index(drop=True)
            ds_sample_messages = HFDataset.from_pandas(df_by_sample_messages)
            samples_messages.append(ds_sample_messages)

        return samples, samples_messages

    def _preprocess_sample(self, sample: dict[str, Any], target_thoughts: bool, split: str) -> dict[str, Any]:
        """
        Preprocess a chat dialogue sample for Hugging Face Transformers model input
        -> Assumes sample is a dict with at least keys ["state", "action", "tools"]
        -> sample["state"] should be a list of message dicts {"role": str, "content": str}
        -> sample["action"] should either be a message dict or a str
        -> sample["tools"] should be a list of tool description dicts
        
        e.g., https://huggingface.co/datasets/mzio/aprm-sft_genthinkact-ENtw_treasure_hunter-GEaprm_qwen3_ap-SE0-REfsc4-ap1-b019
        """
        # Build model_input_ids (tokens for full state-(thought)-action text)
        messages = [{"role": "system", "content": self.system_prompt}]
        # Get state messages
        if self.actions_only:
            state_msgs = [
                # Separate each msg into "think", "act" parts -> only keep the "act" part
                get_thought_and_actions(msg, **self.parse_action_kwargs)[1]
                for msg in sample["state"]
            ]
        else:
            state_msgs = sample["state"]
        messages.extend(state_msgs)

        # Get full and "thought-only" actions
        think_act_msg: dict[str, str] = (
            {"role": "assistant", "content": sample["action"]}
            if isinstance(sample["action"], str)
            else sample["action"]
        )
        if self.actions_only:  # Only keep the "act" part of the action too
            _, think_act_msg = get_thought_and_actions(think_act_msg, **self.parse_action_kwargs)
        think_msg, _ = get_thought_and_actions(think_act_msg, **self.parse_action_kwargs)

        # Get tokens and lengths for (1) state, (2) full model input, (3) state-thought only
        _shared_kwargs = {"enable_thinking": False, "tools": sample["tools"], "tokenize": True}
        _tokenize = partial(self.tokenizer.apply_chat_template, **_shared_kwargs)
        
        state_ids = _tokenize(messages, add_generation_prompt=True)
        state_thought_ids = _tokenize(
            messages + [think_msg],
            add_generation_prompt=False,
            continue_final_message=True,
        )
        # Full model input (state, thought, action)
        model_input_kwargs = {"add_generation_prompt": False, "return_dict": True}
        if target_thoughts:
            # allow us to easily train on past assistant messages
            model_input_kwargs["return_assistant_tokens_mask"] = True
        model_inputs = _tokenize(messages + [think_act_msg], **model_input_kwargs)
        model_input_ids = model_inputs["input_ids"]
        state_len = len(state_ids)
        state_thought_len = len(state_thought_ids)

        # Finally get model training inputs and labels
        labels = copy(model_input_ids)
        if target_thoughts and split == "train":  # compute cross-entropy loss for thought and action tokens
            # labels[:state_len] = [-100] * state_len
            assistant_mask = np.array(model_inputs["assistant_masks"])
            labels = np.array(labels)
            labels[assistant_mask == 0] = -100  # note that this does not include eos tokens
            labels = labels.tolist()
        elif target_thoughts:  # Only evaluate on last assistant message
            labels[:state_len] = [-100] * state_len
        else:  # only compute cross-entropy loss for action tokens
            labels[:state_thought_len] = [-100] * state_thought_len

        weight = sample.get("advantage", 1.0)  # Optionally still use advantage for SFT weighting
        return {
            # Model inputs
            "input_ids": model_input_ids,
            "labels": labels,
            "attention_mask": [1] * len(model_input_ids),
            # Optionally weight loss
            "weight": weight,
            # Metadata for evals
            "timestep": int(sample[self.timestep_name]),
            "sample_id": int(sample[self.sample_id_name]),
        }

    def shuffle(self, seed: int | None = None) -> None:
        """
        Shuffle dataset
        """
        self.datasets[self.split]    = self.datasets[self.split].shuffle(seed=seed)
        self.datasets_rl[self.split] = self.datasets_rl[self.split].shuffle(seed=seed)

    def reset(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
        try_step: int = 0,
        batch_idx: int = 0,
    ) -> ActionLmState:
        """
        Reset environment (loading a new sample)
        -> Assumes this is only for RL
        """
        dataset, dataset_messages = self.datasets_rl[self.split]
        sample_idx_adj = sample_idx % len(dataset)
        ds_trajectory: HFDataset = dataset[sample_idx_adj]
        ds_trajectory_messages: HFDataset = dataset_messages[sample_idx_adj]
        # sample_loader = DataLoader(ds_trajectory, batch_size=1, shuffle=False)
        return ActionLmState(
            trajectory=ds_trajectory,
            trajectory_messages=ds_trajectory_messages,
        )

    def compute_loss(
        self, 
        model: PreTrainedModel,
        batch: dict[str, torch.Tensor],
        ignore_index: int = -100,
        fp32_loss: bool | None = None,
        eval_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss for a batch of model inputs
        -> We compute as a loss per sequence, which can then be averaged (instead of loss over tokens)
        """
        fp32_loss = fp32_loss or self.fp32_loss
        
        weight = batch.get("weight", 1.0)
        device = model.device
        model_inputs = {
            k: v.to(device) for k, v in batch.items()
            if k in ["input_ids", "attention_mask"]
        }
        logits = model(**model_inputs, use_cache=False).logits[:, :-1, :]
        labels = batch["labels"][:, 1:].to(device)

        batch_size, seq_len_m1, vocab_size = logits.shape
        valid = labels != ignore_index
        token_len = valid.sum(dim=-1)

        # Flatten
        flat_logits = logits.reshape(-1, vocab_size)
        flat_labels = labels.reshape(-1)
        flat_valid  = valid.reshape(-1)
        # Indices of labeled tokens only
        valid_indices = flat_valid.nonzero(as_tuple=True)[0]

        logits_v = flat_logits.index_select(0, valid_indices)
        labels_v = flat_labels.index_select(0, valid_indices)

        token_nll = F.cross_entropy(
            logits_v.to(dtype=torch.float32 if fp32_loss else logits.dtype),
            labels_v,  # .to(device),
            reduction="none",
            # ignore_index=ignore_index,
        ).to(dtype=logits.dtype)

        # Sum NLL per sequence in batch
        # Flatten order is (batch-major), so batch_id = idx // Tm1
        batch_id = valid_indices // seq_len_m1                                               # [N_valid]
        nll_sum = torch.zeros(batch_size, device=device, dtype=token_nll.dtype)
        nll_sum.scatter_add_(0, batch_id, token_nll)                             # [B]

        # token_nll = F.cross_entropy(
        #     logits.view(-1, vocab_size).to(dtype=torch.float32 if fp32_loss else logits.dtype),
        #     labels.view(-1).to(device),
        #     reduction="none",
        #     ignore_index=ignore_index,
        # ).reshape(batch_size, seq_len_m1).to(dtype=logits.dtype)

        # nll_sum = token_nll.sum(dim=1)
        loss = nll_sum / token_len.clamp_min(1)  # .to(device)
        ppl = torch.exp(loss).detach().cpu()
        out = {
            "loss": loss * weight.to(device),
            "ppl": ppl,
            "nll_sum": nll_sum.detach().cpu(),
            "token_len": token_len.detach().cpu(),
        }
        if eval_mode:
            out.update({
                "logits_shifted": logits,  # .detach().cpu(),  # shifted already [:, :-1, :]
                "labels_shifted": labels  # .detach().cpu(),  # shifted already [:, 1:]
            })
        return out

    def eval_lm(
        self,
        model: PreTrainedModel,
        sample_idx: int,
        ignore_index: int = -100,
        current_pbar_pos: int = 0,
        fp32_loss: bool | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single rollout via language-modeling metrics on the target tokens
        """
        fp32_loss = fp32_loss or self.fp32_loss
        state = self.reset(sample_idx=sample_idx)
        ds_trajectory: HFDataset = state.trajectory
        eval_loader = DataLoader(ds_trajectory, batch_size=1, shuffle=False)

        rollout_nll_sum = 0.0  # sum over labeld tokens across all steps
        rollout_tok_len = 0
        rollout_step_correct = 0
        rollout_step_total = 0
        rollout_success = 1.0
        longest_success = 0

        rollout_action_nlls: list[float] = []
        rollout_action_ppls: list[float] = []
        rollout_action_corrects: list[float] = []
        rollout_action_timesteps: list[int] = []

        colour = ROYGBIV[sample_idx % len(ROYGBIV)]
        pbar = tqdm(
            # eval_loader, desc=f"Evaluating Rollout {sample_idx} / {len(self) - 1}",
            eval_loader, desc=f"Evaluating Rollout {sample_idx + 1} / {len(self)} via Inference",
            leave=True, colour=colour, position=current_pbar_pos + 1,
        )
        for step_idx, batch in enumerate(pbar):
            metrics = self.compute_loss(model, batch, eval_mode=True, fp32_loss=fp32_loss)

            logits = metrics["logits_shifted"]  # (1, seq_len - 1, vocab_size)
            labels = metrics["labels_shifted"]  # (1, seq_len - 1)
            _valid = labels != ignore_index
            length = _valid.sum(dim=-1)

            # Compute action-correctness
            tok_pred = logits.argmax(dim=-1)    # (1, seq_len - 1)
            _correct = (tok_pred == labels) & _valid
            action_correct = (_correct.sum(dim=-1) == length).float().item()

            rollout_nll_sum += metrics["nll_sum"].item()
            rollout_tok_len += length.item()

            rollout_step_correct += action_correct
            rollout_step_total += 1
            rollout_success *= action_correct
            if rollout_success > 0:
                longest_success += 1

            # Keep track of individual action metrics
            rollout_action_nlls.append(metrics["nll_sum"].item())
            rollout_action_ppls.append(metrics["ppl"].item())
            rollout_action_corrects.append(action_correct)
            rollout_action_timesteps.append(step_idx)

            _step_metrics = {
                k: _get_item(v) for k, v in metrics.items() if not k.endswith("shifted")
            }
            _step_metrics.update({
                "action_correct": _get_item(action_correct),
                "rollout_success": _get_item(int(rollout_success)),
                "longest_success": _get_item(longest_success),
            })
            pbar.set_postfix(**_step_metrics)

            # Visualize samples?
            if sample_idx == 0:
                check_model_inputs(batch, self.tokenizer)

        rollout_nll = rollout_nll_sum / max(rollout_tok_len, 1)
        rollout_ppl = torch.exp(torch.tensor(rollout_nll)).item()

        return {
            # Rollout-wise metrics
            "rollout_nll": rollout_nll,
            "rollout_ppl": rollout_ppl,
            "rollout_step_correct": rollout_step_correct,
            "rollout_step_total": rollout_step_total,
            "rollout_step_acc": rollout_step_correct / max(rollout_step_total, 1),
            "rollout_success": int(rollout_success),
            "longest_success": longest_success,
            # Action-wise metrics
            "rollout_action_nll": rollout_action_nlls,
            "rollout_action_ppl": rollout_action_ppls,
            "rollout_action_correct": rollout_action_corrects,
            "rollout_action_timestep": rollout_action_timesteps,
        }

    def eval_gen(
        self,
        model: PreTrainedModel,
        sample_idx: int,
        current_pbar_pos: int = 0,
        fp32_loss: bool | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single rollout via generation metrics on the target tokens
        """
        fp32_loss = fp32_loss or self.fp32_loss
        state = self.reset(sample_idx=sample_idx)
        ds_trajectory_messages: HFDataset = state.trajectory_messages
        
        rollout_step_correct = 0
        rollout_step_total = 0
        rollout_success = 1.0
        longest_success = 0

        rollout_action_corrects: list[float] = []
        rollout_action_timesteps: list[int] = []

        colour = ROYGBIV[::-1][sample_idx % len(ROYGBIV)]
        pbar = tqdm(
            # eval_loader, desc=f"Evaluating Rollout {sample_idx} / {len(self) - 1}",
            ds_trajectory_messages,
            desc=f"Evaluating Rollout {sample_idx + 1} / {len(self)} via Generation",
            leave=True, colour=colour, position=current_pbar_pos + 1,
        )
        for step_idx, sample in enumerate(pbar):
            # Generate with the model
            device = model.device

            # state = sample["state"]
            messages = [{"role": "system", "content": self.system_prompt}]
            # Get state messages
            if self.actions_only:
                state_msgs = [
                    # Separate each msg into "think", "act" parts -> only keep the "act" part
                    get_thought_and_actions(msg, **self.parse_action_kwargs)[1]
                    for msg in sample["state"]
                ]
            else:
                state_msgs = sample["state"]
            messages.extend(state_msgs)

            state_inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tools=sample["tools"],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_len = state_inputs["input_ids"].shape[1]
            outputs = model.generate(
                **state_inputs.to(device),
                max_new_tokens=1024,
                num_return_sequences=1,  # maybe more if we want to do best of N
                temperature=0.7,
                # do_sample=False,
                streamer=self.streamer,
                pad_token_id=self.tokenizer.eos_token_id,
            ).detach().cpu()
            decoded_texts = self.tokenizer.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            # For now we only allow one tool call per response
            decoded_texts = [
                f"{text.split(self.action_eos)[0]}{self.action_eos}"
                if self.action_eos in text
                else text
                for text in decoded_texts
            ]
            # Extract action from decoded text
            action_preds = [
                text.split(self.action_bos)[-1].split(self.action_eos)[0].strip()
                for text in decoded_texts
            ]
            # Get ground-truth action (a bit hacky, we decode from token ids)
            action_true = sample["target_action"].strip()

            # Compute action-correctness
            action_correct = [action_pred == action_true for action_pred in action_preds]
            action_correct = sum(action_correct) / max(len(action_correct), 1)

            rollout_step_correct += action_correct
            rollout_step_total += 1
            rollout_success *= action_correct

            if rollout_success > 0:
                longest_success += 1

            if self.streamer is not None:
                # _target_act = self.tokenizer.apply_chat_template(
                #     [{"role": "user", "content": action_true}],
                #     add_generation_prompt=False, tools=sample["tools"], tokenize=False,
                # )
                _target_act = f"{self.action_bos}\n{action_true}\n{self.action_eos}"
                _colour = f"bold {"bright_green" if action_correct else "bright_red"}"
                rich_print(f"\n[{_colour}]**Target Action**\n{_target_act}[/{_colour}]")
                rich_print(f"[bold]Run url: [link={self.run_url}]{self.run_url}[/link][/bold]")
                rich_print(f"[bold]Run cmd: [bright_cyan]{self.run_cmd}[/bright_cyan][/bold]")

            pbar.set_postfix({
                # "target_action": action_true,
                "correct": rollout_step_correct,
                "total": rollout_step_total,
                "acc": rollout_step_correct / max(rollout_step_total, 1),
                "success": int(rollout_success),
                "longest_success": longest_success,
            })

            if sample_idx == 0:  # a bit of a hack
                # check_model_inputs(state_inputs, self.tokenizer)
                decoded_inputs = self.tokenizer.batch_decode(state_inputs["input_ids"])[0]
                rich_print(f"[cyan]Input {sample_idx}:\n{decoded_inputs}\n[/cyan]")

        return {
            # Rollout-wise metrics
            "rollout_step_correct": rollout_step_correct,
            "rollout_step_total": rollout_step_total,
            "rollout_step_acc": rollout_step_correct / max(rollout_step_total, 1),
            "rollout_success": int(rollout_success),
            "longest_success": longest_success,
            # Action-wise metrics
            "rollout_action_correct": rollout_action_corrects,
            "rollout_action_timestep": rollout_action_timesteps,
        }
    
    def step(
        self,
        **kwargs: Any,
    # ) -> ActionLmStepResult:
    ) -> None:
        """
        Step through the environment; see `_step_impl` for details
        """
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        current_state: ActionLmState,
        **kwargs: Any,
    # ) -> ActionLmStepResult:
    ) -> None:
        """
        Dummy method for now
        """
        return None
        # return ActionLmStepResult(
        #     state=current_state,
        #     reward=0.0,
        #     done=True,
        #     truncated=False,
        #     info=None,
        # )
