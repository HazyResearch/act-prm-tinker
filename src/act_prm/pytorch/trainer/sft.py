"""
PyTorch trainer for Hugging Face Transformer (PEFT) models
"""

import os
import logging
import sys
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
from ..train import run_rollouts

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


class SftTrainer:
    """
    Supervised Fine-Tuning for Hugging Face Transformer (PEFT) models
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
        self.replay_buffer = replay_buffer
        self.env = env
        self.eval_env = eval_env
        self.ml_logger = ml_logger
        self.hf_tokenizer = hf_tokenizer
        self.fp32_loss = cfg.get("fp32_loss", False)

        self.run_name = cfg.run_name
        self.run_url = ml_logger.get_logger_url() if ml_logger is not None else None
        # self.run_cmd = f"uv run python main.py {" ".join(sys.argv[1:])}"
        self.run_cmd = " ".join(sys.argv)

        # RL / Evaluation generator: does standard rollouts, see act_prm/generator/default.py
        self.rl_generator_constructor = self.get_generator_constructor(
            name="default",
            verbose=generator_cfg.verbose,
        )

        # Checkpointing and best metrics
        self.checkpoint_path = checkpoint_path or cfg.checkpoint_path
        self.best_lm_checkpoint_path = os.path.join(checkpoint_path, "step_best_lm")
        self.best_ro_checkpoint_path = os.path.join(checkpoint_path, "step_best_ro")
        if not os.path.exists(self.best_lm_checkpoint_path):
            os.makedirs(self.best_lm_checkpoint_path)
        if not os.path.exists(self.best_ro_checkpoint_path):
            os.makedirs(self.best_ro_checkpoint_path)
        
        self.best_ppl = float("inf")      # for perplexity evaluation
        self.best_ro_metric = float("-inf")  # for rollout evaluation
        self.best_ppl_step = 0
        self.best_ro_metric_step = 0

        # Save action-wise metrics
        self.eval_lm_data_actions: dict[str, list[int | float]] = {
            "train_step_idx": [],
            "train_eval_idx": [],
            "rollout_task_idx": [],
        }  # fill in the rest of the columns below
        self.eval_gen_data_actions: dict[str, list[int | float]] = {
            "train_step_idx": [],
            "train_eval_idx": [],
            "rollout_task_idx": [],
        }  # fill in the rest of the columns below
        log_path = log_path or cfg.log_path
        self.eval_lm_data_path = os.path.join(log_path, "eval_lm_action_metrics.csv")
        self.eval_gen_data_path = os.path.join(log_path, "eval_gen_action_metrics.csv")

    def get_generator_constructor(self, **kwargs: Any) -> Callable[..., HuggingFaceGenerator]:
        """
        Get a (partially initialized) Hugging Face Generator constructor by name
        """
        return get_generator_constructor(**kwargs, ml_logger=self.ml_logger, cfg=self.cfg)

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
        ignore_index: int = -100,
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
        batch_size, seq_len_m1, vocab_size = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size).to(dtype=torch.float32 if fp32_loss else logits.dtype),
            labels.view(-1).to(device),
            reduction="mean",
            ignore_index=ignore_index,
        ).to(dtype=logits.dtype)
        ppl = torch.exp(loss).detach().cpu()
        return {"loss": loss, "ppl": ppl, "weight": weight}

    def _compute_env_loss(
        self, 
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        ignore_index: int = -100,
        fp32_loss: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss for a batch of model inputs
        """
        fp32_loss = fp32_loss or self.fp32_loss
        weight = batch.get("weight", 1.0)
        device = model.device
        model_inputs = {
            k: v.to(device) for k, v in batch.items()
            if k in ["input_ids", "attention_mask"]
        }
        logits = model(**model_inputs, use_cache=False).logits[:, :-1, :]
        labels = batch["labels"][:, 1:]

        batch_size, seq_len_m1, vocab_size = logits.shape
        valid = labels != ignore_index
        token_len = valid.sum(dim=-1)

        token_nll = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size).to(dtype=torch.float32 if fp32_loss else logits.dtype),
            labels.view(-1).to(device),
            reduction="none",
            ignore_index=ignore_index,
        ).reshape(batch_size, seq_len_m1).to(dtype=logits.dtype)

        nll_sum = token_nll.sum(dim=1)
        loss = nll_sum / token_len.clamp_min(1).to(device)
        ppl = torch.exp(loss).detach().cpu()
        return {
            "loss": loss * weight.to(device),
            "ppl": ppl,
            "nll_sum": nll_sum.detach().cpu(),
            "token_len": token_len.detach().cpu(),
            "logits_shifted": logits.detach().cpu(),  # shifted already [:, :-1, :]
            "labels_shifted": labels.detach().cpu(),  # shifted already [:, 1:]
        }
    
    def train(
        self,
        llm: HuggingFaceLLM | None = None,
        optimizer: Optimizer | Any | None = None,
        cfg: DictConfig | None = None,
        env: Environment | None = None,
        eval_env: Environment | None = None,
        # do_rollout_eval: bool | None = None,
        eval_every: int | None = None,
        eval_gen_every: int | None = None,
        eval_rollout_every: int | None = None,
        # Specify training duration
        num_steps: int | None = None,
        mini_batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,  # 1 if not specified here or in cfg
        num_eval_gen_samples: int | None = None,
        num_eval_rollout_samples: int | None = None,
        # Other identifiers
        checkpoint_name: str | None = None,
        name_or_identifier: str | None = None,
        **kwargs: Any,
    ) -> HuggingFaceLLM:
        """
        Implement entire SFT training loop for Hugging Face Transformer (PEFT) model (llm.model)

        At a high level, we:

        1. Prepare SFT dataloaders from `env.datasets` for each train/eval split
        2. Train for next-token prediction via cross-entropy loss on individual train-split samples
           (typically over thought-action tokens)
        3. Periodically evaluate the model on eval-split samples (typically on action tokens only)
           - If do_rollout_eval is True, we also evaluate on `eval_env` with RL-style rollouts
        """
        llm = llm or self.llm
        optimizer = optimizer or self.optimizer

        cfg = cfg or self.cfg
        env = env or self.env
        eval_env = eval_env or self.eval_env
        # Evaluation
        hf_tokenizer = self.hf_tokenizer
        # Batch iterations to evaluate on
        eval_every = eval_every or cfg.eval_every
        eval_gen_every = eval_gen_every or cfg.eval_gen_every
        eval_rollout_every = eval_rollout_every or cfg.eval_rollout_every
        # Number of offline eval samples to evaluate on
        num_eval_lm_samples = len(env.datasets["eval"])
        num_eval_gen_samples = num_eval_gen_samples or cfg.get("num_eval_gen_samples", 5)
        # Number of online eval samples to evaluate on
        num_eval_rollout_samples = num_eval_rollout_samples or cfg.get("num_eval_rollout_samples", 5)

        # Prepare SFT dataloaders from `env.datasets` for each train/eval split
        dataloaders = {"train": None, "eval": None}

        # 1. Determine mechanical dataset batch size and number of epochs
        num_steps = num_steps or cfg.num_steps
        mini_batch_size = mini_batch_size or cfg.mini_batch_size
        grad_accum_step = gradient_accumulation_steps or cfg.gradient_accumulation_steps or 1
        dataloader_batch_size = mini_batch_size // grad_accum_step

        try:
            assert grad_accum_step <= mini_batch_size, (
                f"gradient_accumulation_steps ({grad_accum_step}) must be "
                "less than or equal to mini_batch_size ({mini_batch_size})"
            )
        except AssertionError as e:
            logger.error(f"AssertionError: {e}")

        # 2. Create dataloaders
        for split in dataloaders.keys():
            dataloaders[split] = DataLoader(
                env.datasets[split],
                batch_size=dataloader_batch_size,
                shuffle=True if split == "train" else False,
            )
        train_loader = dataloaders["train"]
        # eval_loader  = dataloaders["eval"]

        # Do training loop
        pbar = tqdm(total=num_steps, desc="Training steps", colour="cyan", position=1)
        step_idx  = 0  # total number of steps or gradient updates taken
        batch_idx = 0  # current batch index within dataloader
        eval_idx  = 0  # number of times we've evaluated
        eval_already = False  # use to only evaluate once per step_idx
        save_already = False  # use to only save once per step_idx
        loss_metrics = None   # will assign to metrics after each batch
        _global_step = 0
        
        train_iterator = iter(train_loader)  # Loop thru dataloader
        while step_idx < num_steps:
            # Running metrics per batch
            metrics = {"grad_step": step_idx, "global_step": _global_step}  
            # Save model checkpoint
            if not save_already and step_idx % cfg.save_every == 0:
                save_lora(llm.model, f"{self.checkpoint_path}/step_{step_idx:03d}")
                save_already = True

            # Evaluate
            last_step = step_idx == num_steps - 1
            do_lm_eval = (
                eval_every > 0 and (
                    (step_idx + 1) % eval_every == 0
                    or last_step
                    or (step_idx == 0 and not cfg.no_initial_eval)
                )
            )
            do_gen_eval = eval_gen_every > 0 and ((step_idx + 1) % eval_gen_every == 0 or last_step)
            do_rollout_eval = eval_rollout_every > 0 and ((step_idx + 1) % eval_rollout_every == 0 or last_step)
            if not eval_already:
                llm.model.eval()  # redundant but just in case
                env.split = "eval"
                if loss_metrics is not None:
                    display_metrics(loss_metrics, title=f"Train Metrics, Step {step_idx}")

                # Do perplexity-based and inference-based evaluation on the eval set data
                if do_lm_eval:
                    _eval_lm_metrics = self._eval_offline_lm(
                        llm, env, step_idx, eval_idx, num_eval_lm_samples,
                    )
                    eval_lm_metrics, eval_lm_metrics_per_task, eval_lm_data_actions = _eval_lm_metrics
                    metrics.update(eval_lm_metrics)                    
                    pbar.set_postfix(**eval_lm_metrics)
                    display_metrics(eval_lm_metrics, title=f"LM Eval Metrics, Step {step_idx}", style="bright_blue")
                    # Save action-level metrics
                    for k, v in eval_lm_data_actions.items():
                        if k not in self.eval_lm_data_actions:
                            self.eval_lm_data_actions[k] = []
                        self.eval_lm_data_actions[k].extend(v)
                    pd.DataFrame(self.eval_lm_data_actions).to_csv(self.eval_lm_data_path, index=False)
                    logger.info(
                        f"LM EVAL (Step {step_idx}, Eval {eval_idx}): "
                        f"Saved offline LM eval metrics to {self.eval_lm_data_path}!"
                    )

                if do_gen_eval:
                    _eval_gen_metrics = self._eval_offline_gen(
                        llm, env, step_idx, eval_idx, num_eval_gen_samples,
                    )
                    eval_gen_metrics, eval_gen_metrics_per_task, eval_gen_data_actions = _eval_gen_metrics
                    metrics.update(eval_gen_metrics)
                    # pbar.set_postfix(**eval_gen_metrics)
                    display_metrics(eval_gen_metrics, title=f"Gen Eval Metrics, Step {step_idx}", style="bright_green")
                    # Save action-level metrics
                    for k, v in eval_gen_data_actions.items():
                        if k not in self.eval_gen_data_actions:
                            self.eval_gen_data_actions[k] = []
                        self.eval_gen_data_actions[k].extend(v)
                    pd.DataFrame(self.eval_gen_data_actions).to_csv(self.eval_gen_data_path, index=False)
                    logger.info(
                        f"Gen EVAL (Step {step_idx}, Eval {eval_idx}): "
                        f"Saved offline Gen eval metrics to {self.eval_gen_data_path}!"
                    )

                if do_rollout_eval:
                    rollout_checkpoint_name = (
                        f"rollout_eval_{checkpoint_name}" if checkpoint_name else "rollout_eval"
                    )
                    eval_env.split = "eval"
                    eval_rollout_metrics, _ = run_rollouts(
                        llm=llm,
                        hf_tokenizer=hf_tokenizer,
                        generator_constructor=self.rl_generator_constructor,
                        env=eval_env,
                        cfg=cfg,
                        batch_id=batch_idx,
                        checkpoint_name=rollout_checkpoint_name,
                        split="eval",
                        num_tries=cfg.eval_num_tries,
                        # Just use all eval tasks
                        start_idx=0,  
                        tasks_per_update=len(eval_env),
                        name_or_identifier=name_or_identifier,
                    )
                    metrics.update(eval_rollout_metrics)
                    display_metrics(eval_rollout_metrics, title=f"Rollout Eval Metrics, Step {step_idx}", style="bright_red")
                    
                    # _eval_rollout_metrics = self._eval_offline_rollout(
                    #     llm, eval_env, step_idx, eval_idx, self.eval_rollout_data_path, num_eval_rollout_samples
                    # )
                    # eval_rollout_metrics, eval_rollout_metrics_per_task, eval_rollout_data_actions = _eval_rollout_metrics
                        
                eval_already = True
                eval_idx += 1

                del eval_lm_metrics, eval_lm_metrics_per_task, eval_lm_data_actions
                del eval_gen_metrics, eval_gen_metrics_per_task, eval_gen_data_actions
                # del eval_rollout_metrics, eval_rollout_metrics_per_task, eval_rollout_data_actions
                torch.cuda.empty_cache()
            
            # Loop through training batches
            try:
                batch: dict[str, torch.Tensor] = next(train_iterator)
            except StopIteration:
                # If we reach end of iterator (epoch), recreate (PyTorch loader shuffles)
                train_iterator = iter(train_loader)
                batch: dict[str, torch.Tensor] = next(train_iterator)

            # --- Training Update ---
            # Sanity-check model inputs
            if (batch_idx) == 0 or (batch_idx + 1) % 10 == 0:
                self._check_model_inputs(batch, hf_tokenizer, cfg)
                # breakpoint()

            llm.model.train()
            # loss, ppl = self.compute_loss(llm.model, batch, cfg)
            # loss_metrics = env.compute_loss(llm.model, batch, fp32_loss=self.fp32_loss)
            loss_metrics = self.compute_loss(llm.model, batch, fp32_loss=self.fp32_loss)
            loss = loss_metrics["loss"]
            # ppl  = loss_metrics["ppl"]
            loss = loss / grad_accum_step
            loss.backward()
            
            batch_idx += 1
            if (batch_idx) % grad_accum_step == 0:
                # Perform gradient update
                optimizer.step()
                optimizer.zero_grad()
                step_idx += 1
                eval_already = False
                save_already = False
                pbar.update(1)

            loss_metrics = {
                f"train/{k}": v.detach().cpu().item() for k, v in loss_metrics.items()
                if v.numel() == 1  # only keep scalar metrics
            }
            metrics.update(loss_metrics)
            # pbar.update(1)
            pbar.set_postfix(**loss_metrics)

            # Log metrics
            try:
                # self.ml_logger.log_metrics(metrics, step=step_idx)
                self.ml_logger.log_metrics(metrics)  # incremets each time

            except Exception as e:
                _error_class = e.__class__.__name__
                _error_message = str(e)
                rich_print(f"[red]Error logging metrics: {_error_class}: {_error_message}[/red]")
                for k, v in metrics.items():
                    print(k, type(v))
                breakpoint()
            torch.cuda.empty_cache()
            _global_step += 1

        # Load best model checkpoint
        llm.model = load_lora(llm.model, self.best_lm_checkpoint_path)
        return llm


    def _eval_offline_lm(
        self,
        llm: HuggingFaceLLM,
        env: Environment,
        step_idx: int,
        eval_idx: int,  # number of times we've evaluated so far
        num_eval_samples: int | None = None,
    ) -> tuple[dict[str, float], dict[str, list[float]], dict[str, list[Any]]]:
        """
        Do offline language modeling evaluation
        - For given ground-truth trajectory, at each timestep evaluate the model's
          inference-based language modeling over the ground-truth action tokens

        Returns:
        - eval_metrics_to_log: dict[str, float] - metrics for logging via ml_logger
        - eval_metrics_per_task: dict[str, list[float]] - metrics per task
        - eval_data_actions: dict[str, list[Any]] - action-level metrics
        """
        with torch.no_grad():
            llm.model.eval()
            orig_env_split = copy(env.split)
            env.split = "eval"  # ensure we use "eval" samples / tasks
            # Get samples as minimum of len(env) or specified num_eval_samples
            num_eval_samples = num_eval_samples or len(env)
            num_eval_samples = min(num_eval_samples, len(env))

            eval_metrics_per_task: dict[list[float]] = {
                "nll_per_task": [],
                "ppl_per_task": [],
                "step_acc_per_task": [],
                "success_per_task": [],
                "longest_per_task": [],
            }
            eval_data_actions: dict[str, list[Any]] = {"task_idx": []}

            for task_idx in range(num_eval_samples):
                eval_metrics = env.eval_lm(llm.model, task_idx, fp32_loss=self.fp32_loss)
                eval_metrics_per_task["nll_per_task"].append(eval_metrics["rollout_nll"])
                eval_metrics_per_task["ppl_per_task"].append(eval_metrics["rollout_ppl"])
                eval_metrics_per_task["step_acc_per_task"].append(eval_metrics["rollout_step_acc"])
                eval_metrics_per_task["success_per_task"].append(eval_metrics["rollout_success"])
                eval_metrics_per_task["longest_per_task"].append(eval_metrics["longest_success"])

                # Fill in action-level metrics by action timestep
                for k in eval_metrics.keys():
                    if k not in eval_data_actions and k.startswith("rollout_action_"):
                        eval_data_actions[k] = []
                for _t in eval_metrics["rollout_action_timestep"]:
                    eval_data_actions["train_step_idx"].append(step_idx)
                    eval_data_actions["train_eval_idx"].append(eval_idx)
                    eval_data_actions["rollout_task_idx"].append(task_idx)
                    for k in eval_data_actions.keys():
                        eval_data_actions[k].append(eval_metrics[k][_t])
                    eval_data_actions["task_idx"].append(task_idx)  # redundant but sanity check
                # pd.DataFrame(eval_data_actions).to_csv(eval_data_save_path, index=False)
                # logger.info(
                #     f"LM EVAL (Step {step_idx}, Eval {eval_idx}): Saved offline LM eval metrics "
                #     f"for task {task_idx} to {eval_data_save_path}"
                # )

        # Get metrics for logging via ml_logger
        def _get_metric_key(k: str) -> str:
            k = k.replace("_per_task", "")
            return f"eval/lm_{k}"
        eval_metrics_to_log = {
            _get_metric_key(k): sum(v) / max(len(v), 1)
            for k, v in eval_metrics_per_task.items()
            if k not in ["ppl_per_task"]
        }
        # Save specific PPL and action prob metrics
        eval_metrics_to_log.update({
            "eval/lm_ppl": np.exp(eval_metrics_to_log["eval/lm_nll"]).item(),
            "eval/lm_probs": np.exp(-eval_metrics_to_log["eval/lm_nll"]).item(),
        })
        # Update best metrics 
        eval_ppl = eval_metrics_to_log["eval/lm_ppl"]
        if eval_ppl < self.best_ppl:
            self.best_ppl = eval_ppl
            self.best_ppl_step = step_idx
            # torch.save(llm.model.state_dict(), self.best_lm_checkpoint_path)
            save_lora(llm.model, self.best_lm_checkpoint_path)
            logger.info(
                f"LM EVAL (Step {step_idx}, Eval {eval_idx}): "
                f"Updated best LM PPL to {eval_ppl} at step {step_idx}"
            )
        eval_metrics_to_log.update({
            "eval/lm_best_ppl": self.best_ppl, "eval/lm_best_ppl_step": self.best_ppl_step,
        })
        eval_metrics_to_log["actions_data_lm_save_path"] = self.eval_lm_data_path
        # metrics.update(eval_metrics)
        # pbar.set_postfix(**eval_metrics)
        # display_metrics(eval_metrics, title=f"LM Eval Metrics, Step {step_idx}", style="bright_green")            
        env.split = orig_env_split
        return eval_metrics_to_log, eval_metrics_per_task, eval_data_actions


    def _eval_offline_gen(
        self,
        llm: HuggingFaceLLM,
        env: Environment,
        step_idx: int,
        eval_idx: int,  # number of times we've evaluated so far
        num_eval_samples: int | None = None,
    ) -> tuple[dict[str, float], dict[str, list[float]], dict[str, list[Any]]]:
        """
        Do offline generation-based evaluation
        - For given ground-truth trajectory, at each timestep evaluate the model's
          generation from the input state, and compare to the ground-truth action

        Returns:
        - eval_metrics_to_log: dict[str, float] - metrics for logging via ml_logger
        - eval_metrics_per_task: dict[str, list[float]] - metrics per task
        - eval_data_actions: dict[str, list[Any]] - action-level metrics
        """
        # MZ 1/27/26 TODO: Def some code memoization / abstraction to do here
        with torch.no_grad():
            llm.model.eval()
            orig_env_split = copy(env.split)
            env.split = "eval"  # ensure we use "eval" samples / tasks
            # Get samples as minimum of len(env) or specified num_eval_samples
            num_eval_samples = num_eval_samples or len(env)
            num_eval_samples = min(num_eval_samples, len(env))

            eval_metrics_per_task: dict[list[float]] = {
                "step_acc_per_task": [],
                "success_per_task": [],
                "longest_per_task": [],
            }
            eval_data_actions: dict[str, list[Any]] = {"task_idx": []}

            for task_idx in range(num_eval_samples):
                eval_metrics = env.eval_gen(llm.model, task_idx, fp32_loss=self.fp32_loss)
                eval_metrics_per_task["step_acc_per_task"].append(eval_metrics["rollout_step_acc"])
                eval_metrics_per_task["success_per_task"].append(eval_metrics["rollout_success"])
                eval_metrics_per_task["longest_per_task"].append(eval_metrics["longest_success"])

                # Fill in action-level metrics by action timestep
                for k in eval_metrics.keys():
                    if k not in eval_data_actions and k.startswith("rollout_action_"):
                        eval_data_actions[k] = []
                for _t in eval_metrics["rollout_action_timestep"]:
                    eval_data_actions["train_step_idx"].append(step_idx)
                    eval_data_actions["train_eval_idx"].append(eval_idx)
                    eval_data_actions["rollout_task_idx"].append(task_idx)
                    for k in eval_data_actions.keys():
                        eval_data_actions[k].append(eval_metrics[k][_t])
                    eval_data_actions["task_idx"].append(task_idx)  # redundant but sanity check
                # pd.DataFrame(eval_data_actions).to_csv(eval_data_save_path, index=False)
                # logger.info(
                #     f"Gen EVAL (Step {step_idx}, Eval {eval_idx}): Saved offline Gen eval metrics "
                #     f"for task {task_idx} to {eval_data_save_path}"
                # )
        # Get metrics for logging via ml_logger
        def _get_metric_key(k: str) -> str:
            k = k.replace("_per_task", "")
            return f"eval/gen_{k}"
        eval_metrics_to_log = {
            _get_metric_key(k): sum(v) / max(len(v), 1)
            for k, v in eval_metrics_per_task.items()
            if k not in ["ppl_per_task"]
        }
        eval_metrics_to_log["actions_data_gen_save_path"] = self.eval_gen_data_path
        # For now, don't track these metrics to update best metrics 
        # eval_gen_step_acc = eval_metrics_to_log["eval/gen_step_acc"]
        # if eval_gen_step_acc > self.best_gen_step_acc:
        #     self.best_gen_step_acc = eval_gen_step_acc
        #     self.best_gen_step_acc_step = step_idx
        #     save_lora(llm.model, self.best_gen_checkpoint_path)
        #     logger.info(
        #         f"Gen EVAL (Step {step_idx}, Eval {eval_idx}): "
        #         f"Updated best Gen step accuracy to {eval_gen_step_acc} at step {step_idx}"
        #     )
        # eval_metrics_to_log.update({
        #     "eval/gen_best_step_acc": self.best_gen_step_acc, "eval/gen_best_step_acc_step": self.best_gen_step_acc_step,
        # })
        env.split = orig_env_split
        return eval_metrics_to_log, eval_metrics_per_task, eval_data_actions
