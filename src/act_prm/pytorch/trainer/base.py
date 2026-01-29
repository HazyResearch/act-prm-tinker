"""
Parent class PyTorch trainer for Hugging Face Transformers models
"""

import os
import logging
import sys
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import Any, Callable

from omegaconf import DictConfig
from rich import print as rich_print
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from transformers import PreTrainedTokenizerBase
from tinker_cookbook.utils import ml_log

from act_prm.lora import load_lora, save_lora
from act_prm.llm_handlers import HuggingFaceLLM
from act_prm.environments import Environment
from act_prm.replay_buffer import ReplayBuffer

from ..generator import get_generator_constructor, HuggingFaceGenerator
from ..train import run_rollouts, hide_observations, prepare_minibatch

logger = logging.getLogger(__name__)
console = Console()


def display_metrics(
    metrics: dict[str, Any],
    title: str | None = None,
    style: str = "bright_yellow",
) -> None:
    """
    Display metrics in a table
    """
    table = Table(title=title, style=style)
    table.add_column("Metric", justify="left", style=style)
    table.add_column("Value", justify="left", style=f"bold {style}")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


class BaseTrainer(ABC):
    """
    Parent class for PyTorch trainers (Hugging Face Transformers models)
    """
    def __init__(
        self,
        cfg: DictConfig,
        llm: HuggingFaceLLM,
        optimizer: Optimizer | Any,  # or a HF one?
        generator_cfg: DictConfig,
        replay_buffer: ReplayBuffer,
        env: Environment,
        eval_env: Environment,
        ml_logger: ml_log.Logger,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        checkpoint_path: str | None = None,
        log_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.optimizer = optimizer
        self.generator_cfg = generator_cfg
        self.replay_buffer = replay_buffer
        self.env = env
        self.eval_env = eval_env
        self.ml_logger = ml_logger
        self.hf_tokenizer = hf_tokenizer
        self.fp32_loss = cfg.get("fp32_loss", False)

        # If True, we hide observations other than the last one to avoid context blow-up
        # -> See hide_observations() in ../train.py for more details
        self.hide_observations = cfg.get("hide_observations", False)
        self.hidden_obs_content = cfg.get("hidden_obs_content", "...")

        # RL / Evaluation generator: does standard rollouts, see act_prm/generator/default.py
        self.rl_generator_constructor = self.get_generator_constructor(**self.generator_cfg)
        self.best_metric = 1e8 if "loss" in cfg.best_metric else -1e8

        self.run_name = cfg.run_name
        self.run_url = ml_logger.get_logger_url() if ml_logger is not None else None
        # self.run_cmd = f"uv run python main.py {" ".join(sys.argv[1:])}"
        self.run_cmd = " ".join(sys.argv)

        # Logging and checkpointing
        self.checkpoint_path = checkpoint_path or cfg.checkpoint_path
        self.log_path = log_path or cfg.log_path

    def get_generator_constructor(self, **kwargs: Any) -> Callable[..., HuggingFaceGenerator]:
        """
        Get a (partially initialized) Hugging Face Generator constructor by name
        """
        return get_generator_constructor(**kwargs, ml_logger=self.ml_logger, cfg=self.cfg)

    def maybe_hide_observations(
        self,
        messages: list[dict[str, str]],
        hidden_obs_content: str | None = None,
        first_obs_to_show: int = 2,  # e.g., to keep prompt
        last_obs_to_show: int = 1,   # e.g., to keep last observation
    ) -> list[dict[str, str]]:
        """
        Maybe hide past observations from messages
        """
        if not self.hide_observations:
            return messages

        hidden_obs_content = hidden_obs_content or self.hidden_obs_content
        return hide_observations(messages, hidden_obs_content, first_obs_to_show, last_obs_to_show)

    def prepare_minibatch(self, **kwargs: Any) -> tuple[DataLoader, dict[str, Any]]:
        """
        Prepare a minibatch of trajectories for training
        """
        return prepare_minibatch(**kwargs)

    def _check_model_inputs(
        self,
        batch: dict[str, torch.Tensor],
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        cfg: DictConfig | None = None,
    ) -> None:
        """
        Sanity-check model inputs by rich printing them
        """
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        cfg = cfg or self.cfg

        decoded_inputs = hf_tokenizer.batch_decode(batch["input_ids"][:, 1:])
        _labels = deepcopy(batch["labels"][:, 1:])
        _labels[_labels == -100] = 0  # -100 will cause tokenization errors
        decoded_labels = hf_tokenizer.batch_decode(_labels)
        for idx, decoded_input in enumerate(decoded_inputs):
            rich_print(f"[cyan]Input {idx}:\n{decoded_input}\n[/cyan]")
            rich_print(f"[green]Label {idx}:\n{decoded_labels[idx]}\n[/green]")
            rich_print("=" * 100)
        # Keep run url and cmd in display
        rich_print(f"[bold]Run url: [link={self.run_url}]{self.run_url}[/link][/bold]")
        rich_print(f"[bold]Run cmd: [bright_cyan]{self.run_cmd}[/bright_cyan][/bold]")

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        fp32_loss: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss for a batch of model inputs
        -> Simple mini-batch cross-entropy loss (no weights)
        """
        fp32_loss = fp32_loss or self.fp32_loss
        device = model.device
        model_inputs = {
            k: v.to(device) for k, v in batch.items()
            if k in ["input_ids", "attention_mask"]
        }
        weight = batch.get("weight", 1.0)  # ignored but report it
        logits = model(**model_inputs, use_cache=False).logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        vocab_size = logits.shape[-1]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size).to(dtype=torch.float32 if fp32_loss else logits.dtype),
            labels.view(-1).to(device),
            reduction="mean",
        ).to(dtype=logits.dtype)
        ppl = torch.exp(loss).detach().cpu()
        return {"loss": loss, "ppl": ppl, "weight": weight}
    
    @abstractmethod
    def train(
        self,
        llm: HuggingFaceLLM | None = None,
        optimizer: Optimizer | Any | None = None,
        cfg: DictConfig | None = None,
        env: Environment | None = None,
        eval_env: Environment | None = None,
        eval_every: int | None = None,
        # eval_gen_every: int | None = None,
        # eval_rollout_every: int | None = None,
        # Specify training duration
        num_steps: int | None = None,
        mini_batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,  # 1 if not specified here or in cfg
        # num_eval_gen_samples: int | None = None,
        # num_eval_rollout_samples: int | None = None,
        # Other identifiers
        checkpoint_name: str | None = None,
        name_or_identifier: str | None = None,
        **kwargs: Any,
    ) -> HuggingFaceLLM:
        """
        Implement entire training loop
        """
        raise NotImplementedError
