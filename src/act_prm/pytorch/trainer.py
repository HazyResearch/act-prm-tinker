"""
PyTorch trainer for Hugging Face Transformer (PEFT) models
"""

import os
import sys
from typing import Any

from omegaconf import DictConfig
from rich import print as rich_print
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tinker_cookbook import ml_log

from act_prm.lora import load_lora, save_lora
from act_prm.llm_handlers import HuggingFaceLLM
from act_prm.environments import Environment

from .generator import run_rollouts

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
    table.add_column("Metric", justify="right", style="magenta")
    table.add_column("", justify="left")
    for k, v in metrics.items():
        table.add_row(k, str(v))
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
        replay_buffer: ReplayBuffer,
        env: Environment,
        eval_env: Environment,
        ml_logger: ml_log.Logger,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        checkpoint_path: str | None = None,
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

        self.run_name = cfg.run_name
        self.run_url = ml_logger.get_logger_url() if ml_logger is not None else None
        self.cmd_str = f"uv run python main.py {" ".join(sys.argv[1:])}"

        # Checkpointing and best metrics
        self.checkpoint_path = checkpoint_path
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
        _labels = copy(batch["labels"][:, 1:])
        _labels[_labels == -100] = 0  # -100 will cause tokenization errors
        decoded_labels = hf_tokenizer.batch_decode(_labels)
        for idx, decoded_input in enumerate(decoded_inputs):
            rich_print(f"[cyan]Input {idx}:\n{decoded_input}\n[/cyan]")
            rich_print(f"[green]Label {idx}:\n{decoded_labels[idx]}\n[/green]")
            rich_print("=" * 100)
        # Keep run url and cmd in display
        rich_print(f"[bold]Run url:[link={self.run_url}]{self.run_url}[/link][/bold]")
        rich_print(f"[bold]Run cmd:[bright_blue]{self.run_url}[/bright_blue][/bold]")
    
    def train(
        self,
        # start_batch: int,
        # end_batch: int,
        llm: HuggingFaceLLM | None = None,
        optimizer: Optimizer | Any | None = None,
        cfg: DictConfig | None = None,
        env: Environment | None = None,
        eval_env: Environment | None = None,
        do_rollout_eval: bool | None = None,
        eval_every: int | None = None,
        # Specify training duration
        num_steps: int | None = None,
        mini_batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,  # 1 if not specified here or in cfg
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
        eval_every = eval_every or cfg.eval_every
        do_rollout_eval = do_rollout_eval or cfg.do_rollout_eval

        # Prepare SFT dataloaders from `env.datasets` for each train/eval split
        dataloaders = {"train": None, "eval": None}

        # 1. Determine mechanical dataset batch size and number of epochs
        num_steps = num_steps or cfg.num_steps
        mini_batch_size = mini_batch_size or cfg.mini_batch_size
        grad_accum_step = gradient_accumulation_steps or cfg.gradient_accumulation_steps or 1
        dataloader_batch_size = mini_batch_size // grad_accum_step

        # 2. Create dataloaders
        for split in dataloaders.keys():
            dataloaders[split] = DataLoader(
                env.datasets[split],
                batch_size=dataloader_batch_size,
                shuffle=True if split == "train" else False,
            )
        train_loader = dataloaders["train"]
        eval_loader  = dataloaders["eval"]

        # Do training loop
        pbar = tqdm(total=num_steps, desc="Training steps", colour="cyan", position=1)
        step_idx  = 0  # total number of steps or gradient updates taken
        batch_idx = 0  # current batch index within dataloader
        eval_already = False  # use to only evaluate once per step_idx
        save_already = False  # use to only save once per step_idx
        loss_metrics = None   # will assign to metrics after each batch
        
        train_iterator = iter(train_loader)  # Loop thru dataloader
        while step_idx < num_steps:
            metrics = {}  # Running metrics per batch
            # Save model checkpoint
            if not save_already and step_idx % cfg.save_every == 0:
                save_lora(llm.model, f"{self.checkpoint_path}/step_{step_idx:03d}")
                save_already = True

            # Evaluate 
            if (
                not eval_already and eval_every > 0
                and (step_idx % eval_every == 0 or step_idx == num_steps - 1 or step_idx == 0)
            ):
                if loss_metrics is not None:
                    display_metrics(loss_metrics, title=f"Train Metrics, Step {step_idx}")
                
                # Do perplexity-based and inference-based evaluation on the eval set data
                with torch.no_grad():
                    llm.model.eval()
                    env.split = "eval"
                    num_eval_samples = len(env)
                    # Each eval sample corresponds to a single task, and 
                    # consists of a Dataset of steps in a single task rollout
                    nll_per_task: list[float] = []
                    ppl_per_task: list[float] = []
                    step_acc_per_task: list[float] = []
                    success_per_task: list[float] = []
                    
                    for task_idx in range(num_eval_samples):
                        eval_metrics = env.eval_rollout(llm.model, task_idx)
                        nll_per_task.append(eval_metrics["rollout_nll"])
                        ppl_per_task.append(eval_metrics["rollout_ppl"])
                        step_acc_per_task.append(eval_metrics["rollout_step_acc"])
                        success_per_task.append(eval_metrics["rollout_success"])
                        
                        # MZ 1/23/26 TODO: Figure out how best to save action-wise metrics

                    eval_nll = sum(nll_per_task) / max(len(nll_per_task), 1)
                    eval_ppl = np.exp(eval_nll).item()
                    eval_probs = np.exp(-eval_nll).item()
                    eval_step_acc = sum(step_acc_per_task) / max(len(step_acc_per_task), 1)
                    eval_success  = sum(success_per_task) / max(len(success_per_task), 1)

                    eval_metrics = {
                        "eval_nll": eval_nll,
                        "eval_ppl": eval_ppl,
                        "eval_probs": eval_probs,
                        "eval_step_act_acc": eval_step_acc,
                        "eval_task_success": eval_success,
                    }
                    metrics.update(eval_metrics)
                    pbar.set_postfix(**eval_metrics)
                    display_metrics(eval_metrics, title=f"LM Eval Metrics, Step {step_idx}", style="bright_green")
                    
                    eval_ppl = eval_metrics["eval_ppl"]
                    if eval_ppl < self.best_ppl:
                        self.best_ppl = eval_ppl
                        self.best_ppl_step = step_idx
                        torch.save(llm.model.state_dict(), self.best_lm_checkpoint_path)
                    eval_metrics.update({
                        "best_ppl": self.best_ppl, "best_ppl_step": self.best_ppl_step,
                    })
                    display_metrics(eval_metrics, title=f"LM Eval Metrics, Step {step_idx}", style="bright_green")

                    # Do rollout or sampling-based evaluation on the eval_env
                    if do_rollout_eval:
                        raise NotImplementedError("Rollout evaluation not implemented yet")
                        # eval_metrics, eval_trajectories = run_rollouts(
                        #     llm,
                        #     hf_tokenizer=hf_tokenizer,
                        #     env=eval_env,
                        #     cfg=cfg,
                        #     batch_id=step_idx,
                        #     split="eval",
                        #     num_tries=cfg.eval_num_tries,
                        #     # Just use all eval tasks
                        #     start_idx=0, 
                        #     tasks_per_update=len(eval_env),
                        #     name_or_identifier=f"eval_{step_idx}",
                        # )
                        # metrics.update(eval_metrics)
                        # # Update best metrics
                        # ro_metric_avg = eval_metrics[cfg.eval_metric]
                        # if ro_metric_avg >= self.best_ro_metric:
                        #     self.best_ro_metric = ro_metric_avg
                        #     self.best_ro_metric_step = step_idx
                        #     save_lora(llm.model, self.best_ro_checkpoint_path)
                        # eval_metrics.update({
                        #     "best_ro_metric": self.best_ro_metric, "best_ro_metric_step": self.best_ro_metric_step,
                        # })
                        # display_metrics(eval_metrics, title=f"RO Eval Metrics, Step {step_idx}", style="bright_red")
                eval_already = True
            
            # Loop through training batches
            try:
                batch: dict[str, torch.Tensor] = next(train_iterator)
            except StopIteration:
                # If we reach end of iterator (epoch), recreate (PyTorch loader shuffles)
                train_iterator = iter(train_loader)
                batch: dict[str, torch.Tensor] = next(train_iterator)

            llm.model.train()
            # loss, ppl = self.compute_loss(llm.model, batch, cfg)
            metrics = env.compute_loss(llm.model, batch, cfg)
            loss = metrics["loss"]
            ppl  = metrics["ppl"]
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

            loss_metrics = {"loss": loss.item(), "ppl": ppl.item()}
            metrics.update(loss_metrics)

            # Sanity-check model inputs
            if (batch_idx) == 1 or (batch_idx) % 100 == 0:
                self._check_model_inputs(batch, hf_tokenizer, cfg)
            
            pbar.update(1)
            pbar.set_postfix(**metrics)

            # Log metrics
            self.ml_logger.log_metrics(metrics, step=step_idx)
            torch.cuda.empty_cache()

        # Load best model checkpoint
        llm.model = load_lora(llm.model, self.best_lm_checkpoint_path)
        return llm
