"""
Supervised Fine-Tuning "Environment"

For now, we just use a way to load Hugging Face datasets and preprocess them for SFT 
"""

import logging
from copy import copy
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset, DatasetDict, load_dataset
from tqdm import tqdm

from ...llm_handlers import ActionFromLLM

from ..act_prm.utils import get_thought_and_actions
from ..base import Environment
from ..types import EnvironmentState, EnvironmentStepResult


logger = logging.getLogger(__name__)


class ActionLmState(EnvironmentState):
    """
    State for Action Language Modeling tasks (Pydantic object)
    """
    trajectory: HFDataset


class ActionLmStepResult(EnvironmentStepResult):
    """
    Step result for HotpotQA multiple choice tasks
    """
    state: ActionLmState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


class ActionLmEnvironment(Environment):
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
        action_bos: str = "<tool_call>",
        action_eos: str = "</tool_call>",
        final_answer_bos: str = "Final Answer: ",
        frac_train_tasks: int = 0.8,
        frac_eval_tasks: int = 0.2,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = "You are a helpful assistant.",
        pretrained_model_config: dict[str, Any] | None = None,
        sample_id_name: str = "unique_data_sample_id",
        timestep_name: str = "timestep",
        generation_id_name: str = "generation_id",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_config = dataset_config
        self.actions_only = actions_only
        self.build_full_states = build_full_states
        self.best_actions_only = best_actions_only
        self.best_halves_only = best_halves_only

        self.sample_id_name = sample_id_name
        self.timestep_name = timestep_name
        self.generation_id_name = generation_id_name
        
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

        # Get tokenizer
        self.pretrained_model_config = pretrained_model_config
        self.tokenizer = self._init_tokenizer()  # see act_prm/environments/base.py
        
        # Load data
        self.system_prompt = system_prompt
        # We'll use the DataFrames for trajectory-specific evaluation
        self.datasets, self.df_train, self.df_eval = self.init_data()
        self.datasets_rl = {
            "train": self.init_rl_data(self.df_train, split="train"),
            "eval":  self.init_rl_data(self.df_eval,  split="eval"),
        }
        
    def __len__(self) -> int:
        """
        Load size of RL dataset split
        """
        # return len(self.datasets[self.split])
        return len(self.datasets_rl[self.split])

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
        if not self.build_full_states:
            return df

        unique_sample_ids = df[self.sample_id_name].unique()
        dfs_by_sample_update = []  # keep track of updated dfs for each sample here (heinous copy)
        
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
                    last_obs: dict[str, str] = state[-1]
                    all_full_states[gen_idx].append(last_obs)  # add latest obs, action
                    all_full_states[gen_idx].append(df_by_sample_step["action"][gen_idx])
                # Update the state for all samples in df_by_sample_step (omit last action)
                # df_by_sample_step["state_full_obs"] = [_state[:-1] for _state in all_full_states]
                df_by_sample_step["state"] = [_state[:-1] for _state in all_full_states]
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

        # We want to process this datasets into sequences of trajectories, also their timesteps,
        # -> So that we can then split them into train and eval splits based on the tasks
        df = ds.to_pandas()  # easier to work with
        df = self._maybe_build_full_states(df)

        # Filter for samples corresponding to best actions or best-half of actions
        if self.best_actions_only:
            df = df[df["best_action"]]
        elif self.best_halves_only:
            df = df[df["best_half"]]

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
            partial(self._preprocess_sample, target_thoughts=True),   # labels include thoughts
            remove_columns=ds_train.column_names,
            load_from_cache_file=False,
        )
        ds_eval = ds_eval.map(
            partial(self._preprocess_sample, target_thoughts=False),  # labels include actions only
            remove_columns=ds_eval.column_names,
            load_from_cache_file=False,
        )
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
        unique_sample_ids = df[self.sample_id_name].unique()
        target_thoughts = True if split == "train" else False

        for _sample_id in unique_sample_ids:
            df_by_sample = df[df[self.sample_id_name] == _sample_id]
            df_by_sample = df_by_sample[df_by_sample["best_action"]]  # only keep one trajectory
            df_by_sample.sort_values(by=self.timestep_name, ascending=True, inplace=True)
            ds_sample = HFDataset.from_pandas(df_by_sample)
            ds_sample = ds_sample.map(
                partial(self._preprocess_sample, target_thoughts=target_thoughts),
                remove_columns=ds_sample.column_names,
                load_from_cache_file=False,
            )
            samples.append(ds_sample.with_format("torch"))

        return samples

    def _preprocess_sample(self, sample: dict[str, Any], target_thoughts: bool) -> dict[str, Any]:
        """
        Preprocess a chat dialogue sample for Hugging Face Transformers model input
        -> Assumes sample is a dict with at least keys ["state", "action", "tools"]
        -> sample["state"] should be a list of message dicts {"role": str, "content": str}
        -> sample["action"] should either be a message dict or a str
        -> sample["tools"] should be a list of tool description dicts
        
        e.g., https://huggingface.co/datasets/mzio/aprm-sft_genthinkact-ENtw_treasure_hunter-GEaprm_qwen3_ap-SE0-REfsc4-ap1-b019
        """
        # Build model_input_ids (tokens for full state-(thought)-action text)
        messages = {"role": "system", "content": self.system_prompt}
        # Get state messages
        if self.actions_only:
            state_msgs = [
                # Separate each msg into "think", "act" parts -> only take "act" part
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
        think_msg, _ = get_thought_and_actions(think_act_msg, **self.parse_action_kwargs)

        # Get tokens and lengths for (1) state, (2) full model input, (3) state-thought only
        _shared_kwargs = {"enable_thinking": False, "tools": sample["tools"], "tokenize": True}
        _tokenize = partial(self.tokenizer.apply_chat_template, **_shared_kwargs)
        
        state_ids = _tokenize(messages, add_generation_prompt=True)
        state_thought_ids = _tokenize(
            messages + [think_msg],
            add_generation_prompt=False,
            continue_final_message=False,
        )
        model_input_ids = self.tokenizer.apply_chat_template(
            messages + [think_act_msg],  # Full model input
            add_generation_prompt=False,
        )
        state_len = len(state_ids)
        state_thought_len = len(state_thought_ids)

        # Finally get model training inputs and labels
        labels = copy(model_input_ids)
        if target_thoughts:  # compute cross-entropy loss for thought and action tokens
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
        dataset = self.datasets_rl[self.split]
        sample_idx_adj = sample_idx % len(dataset)
        ds_trajectory: HFDataset = dataset[sample_idx_adj]
        # sample_loader = DataLoader(ds_trajectory, batch_size=1, shuffle=False)

        return ActionLmState(trajectory=ds_trajectory)

    def compute_loss(
        self, 
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        ignore_index: int = -100,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss for a batch of model inputs
        """
        weight = batch.get("weight", 1.0)
        model_inputs = {
            k: v.to(model.device) for k, v in batch.items()
            if k in ["input_ids", "attention_mask"]
        }
        logits = model(**model_inputs, use_cache=False).logits[:, :-1, :]
        labels = batch["labels"][:, 1:]

        batch_size, seq_len_m1, vocab_size = logits.shape
        device = logits.device

        valid = labels != ignore_index
        token_len = valid.sum(dim=-1)

        token_nll = F.cross_entropy(
            logits.view(-1, vocab_size).float(),
            labels.view(-1).to(device),
            reduction="none",
            ignore_index=ignore_index,
        ).reshape(batch_size, seq_len_m1).to(dtype=logits.dtype)

        nll_sum = token_nll.sum(dim=1)
        loss = nll_sum / token_len.clamp_min(1)
        ppl = torch.exp(loss)
        return {
            "loss": loss * weight,
            "ppl": ppl.detach().cpu(),
            "nll_sum": nll_sum.detach().cpu(),
            "token_len": token_len.detach().cpu(),
            "logits": logits.detach().cpu(),  # shifted already [:, :-1, :]
            "labels": labels.detach().cpu(),  # shifted already [:, 1:]
        }

    def eval_rollout(
        self,
        model: nn.Module,
        sample_idx: int,
        ignore_index: int = -100,
    ) -> dict[str, Any]:
        """
        Evaluate a single rollout
        """
        state = self.reset(sample_idx=sample_idx)
        ds_trajectory: HFDataset = state.trajectory
        eval_loader = DataLoader(ds_trajectory, batch_size=1, shuffle=False)

        rollout_nll_sum = 0.0  # sum over labeld tokens across all steps
        rollout_tok_len = 0
        rollout_step_correct = 0
        rollout_step_total = 0
        rollout_success = 1.0

        rollout_action_nlls: list[float] = []
        rollout_action_ppls: list[float] = []
        rollout_action_corrects: list[float] = []
        rollout_action_timesteps: list[int] = []

        pbar = tqdm(
            eval_loader, desc="Evaluating Rollout {sample_idx}",
            leave=False, colour="yellow", position=2,
        )
        for step_idx, batch in enumerate(pbar):
            metrics = self.compute_loss(model, batch)

            logits = metrics["logits"]
            labels = metrics["labels"]
            _valid = labels != ignore_index
            length = _valid.sum(dim=-1)

            # Compute action-correctness
            tok_pred = logits.argmax(dim=-1)  # 1, seq_len - 1
            _correct = (tok_pred == labels) & _valid
            action_correct = (_correct.sum(dim=-1) == length).float().item()

            rollout_nll_sum += metrics["nll_sum"].item()
            rollout_tok_len += length.item()

            rollout_step_correct += action_correct
            rollout_step_total += 1
            rollout_success *= action_correct

            # Keep track of individual action metrics
            rollout_action_nlls.append(metrics["nll_sum"].item())
            rollout_action_ppls.append(metrics["ppl"].item())
            rollout_action_corrects.append(action_correct)
            rollout_action_timesteps.append(step_idx)

        rollout_nll = rollout_nll_sum / rollout_tok_len.clamp_min(1)
        rollout_ppl = torch.exp(torch.tensor(rollout_nll)).item()

        return {
            # Rollout-wise metrics
            "rollout_nll": rollout_nll,
            "rollout_ppl": rollout_ppl,
            "rollout_step_correct": rollout_step_correct,
            "rollout_step_total": rollout_step_total,
            "rollout_step_acc": rollout_step_correct / max(rollout_step_total, 1),
            "rollout_success": rollout_success,
            # Action-wise metrics
            "rollout_action_nlls": rollout_action_nlls,
            "rollout_action_ppls": rollout_action_ppls,
            "rollout_action_corrects": rollout_action_corrects,
            "rollout_action_timesteps": rollout_action_timesteps,
        }
    
    def step(
        self,
        **kwargs: Any,
    ) -> ActionLmStepResult:
        """
        Step through the environment; see `_step_impl` for details
        """
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        current_state: ActionLmState,
        **kwargs: Any,
    ) -> ActionLmStepResult:
        """
        Dummy method for now
        """
        return ActionLmStepResult(
            state=current_state,
            reward=0.0,
            done=True,
            truncated=False,
            info=None,
        )
