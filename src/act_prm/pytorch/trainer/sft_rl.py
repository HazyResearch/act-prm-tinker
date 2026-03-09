"""
PyTorch SFT-warmup + RL Trainer

Extends RLTrainer with an initial SFT warmup phase: trains on static
supervised data (cross-entropy loss) for `sft_warmup_steps` gradient
updates before switching to the standard RL policy gradient loop.

This mirrors the eval_rollout_start pattern in SftTrainer but applies
it to the training loop itself: SFT warmup provides a good initialization
before on-policy RL sampling begins.
"""

import logging
import os
import time
from os.path import join
from typing import Any

from omegaconf import DictConfig
from rich import print as rich_print
from tqdm import tqdm

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from act_prm.lora import save_lora
from act_prm.llm_handlers import HuggingFaceLLM
from act_prm.environments import Environment

from act_prm.trainer.tinker.utils import timed

from ..train import run_rollouts
from .base import BaseTrainer, display_metrics
from .rl import get_item, is_better, RLTrainer


logger = logging.getLogger(__name__)


class SftRlTrainer(RLTrainer):
    """
    Two-phase trainer: SFT warmup followed by RL policy gradient.

    Phase 1 (SFT warmup):
        - Trains on static data from env.datasets["train"] using cross-entropy loss
        - Runs for `sft_warmup_steps` gradient updates
        - Optionally evaluates via RL rollouts every `sft_eval_every` steps
        - Saves checkpoints every `sft_save_every` steps

    Phase 2 (RL):
        - Standard RLTrainer.train() loop with importance-sampled policy gradient
        - Model and optimizer carry state forward from SFT warmup
    """

    def __init__(
        self,
        cfg: DictConfig,
        checkpoint_path: str | None = None,
        log_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            **kwargs,
        )

        # SFT warmup config
        self.sft_warmup_steps = cfg.get("sft_warmup_steps", 0)
        self.sft_mini_batch_size = cfg.get("sft_mini_batch_size", 1)
        self.sft_gradient_accumulation_steps = cfg.get(
            "sft_gradient_accumulation_steps", 1
        )
        self.sft_eval_every = cfg.get("sft_eval_every", cfg.eval_every)
        self.sft_save_every = cfg.get("sft_save_every", cfg.get("save_every", 50))

        # SFT warmup checkpointing
        self.sft_best_checkpoint_path = join(self.checkpoint_path, "step_best_sft")
        os.makedirs(self.sft_best_checkpoint_path, exist_ok=True)
        self.sft_best_ppl = float("inf")
        self.sft_best_ppl_step = -1

    def _compute_sft_loss(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Cross-entropy loss for SFT warmup (delegates to BaseTrainer)."""
        return BaseTrainer.compute_loss(self, model, batch)

    def sft_warmup(
        self,
        llm: HuggingFaceLLM | None = None,
        optimizer: Optimizer | Any | None = None,
        cfg: DictConfig | None = None,
        env: Environment | None = None,
        eval_env: Environment | None = None,
        checkpoint_name: str | None = None,
        name_or_identifier: str | None = None,
    ) -> None:
        """
        Run SFT warmup phase: cross-entropy training on env.datasets["train"].
        """
        llm = llm or self.llm
        optimizer = optimizer or self.optimizer
        cfg = cfg or self.cfg
        env = env or self.env
        eval_env = eval_env or self.eval_env
        hf_tokenizer = self.hf_tokenizer

        num_steps = self.sft_warmup_steps
        grad_accum_step = self.sft_gradient_accumulation_steps
        dataloader_batch_size = max(1, self.sft_mini_batch_size // grad_accum_step)

        logger.info(
            "SFT warmup: %d steps, batch_size=%d, grad_accum=%d",
            num_steps,
            dataloader_batch_size,
            grad_accum_step,
        )

        # Build DataLoader over static SFT data
        env.split = "train"
        train_loader = DataLoader(
            env.datasets["train"],
            batch_size=dataloader_batch_size,
            shuffle=True,
        )
        train_iterator = iter(train_loader)

        pbar = tqdm(
            total=num_steps, desc="SFT warmup steps", colour="green", position=1
        )
        step_idx = 0
        batch_idx = 0
        eval_already = False
        save_already = False
        loss_metrics = None

        while step_idx < num_steps:
            metrics = {
                "progress/phase": 0,  # 0 = SFT warmup, 1 = RL
                "progress/sft_step": step_idx,
                "progress/sft_done_frac": (step_idx + 1) / num_steps,
                "optim/lr": cfg.learning_rate,
            }
            t_start = time.time()

            # Save checkpoint
            if not save_already and step_idx % self.sft_save_every == 0:
                save_lora(llm.model, f"{self.checkpoint_path}/sft_step_{step_idx:03d}")
                save_already = True

            # Evaluate via RL rollouts (same as RLTrainer eval)
            last_step = step_idx == num_steps - 1
            do_eval = self.sft_eval_every > 0 and (
                (step_idx > 0 and step_idx % self.sft_eval_every == 0) or last_step
            )

            if do_eval and not eval_already:
                with timed("sft_warmup_eval", metrics):
                    eval_env.split = "eval"
                    llm.model.eval()
                    eval_rollout_metrics, _ = run_rollouts(
                        llm=llm,
                        hf_tokenizer=hf_tokenizer,
                        generator_constructor=self.rl_generator_constructor,
                        env=eval_env,
                        cfg=cfg,
                        batch_id=step_idx,
                        checkpoint_name=checkpoint_name,
                        split="eval",
                        num_tries=cfg.eval_num_tries,
                        start_idx=0,
                        tasks_per_update=len(eval_env),
                        name_or_identifier=name_or_identifier,
                    )
                    # Prefix with sft_warmup_eval/
                    prefixed = {
                        f"sft_warmup_eval/{k.split('/')[-1] if '/' in k else k}": v
                        for k, v in eval_rollout_metrics.items()
                    }
                    metrics.update(prefixed)
                    display_metrics(
                        eval_rollout_metrics,
                        title=f"SFT Warmup Eval, Step {step_idx}",
                        style="bright_green",
                    )

                # Update best RL metric (carries forward into RL phase)
                best_metric_key = [
                    k for k in eval_rollout_metrics.keys() if self.best_metric_name in k
                ][0]
                last_metric = eval_rollout_metrics[best_metric_key]
                if is_better(last_metric, self.best_metric, self.best_metric_name):
                    self.best_metric = last_metric
                    self.best_metric_step = step_idx
                    save_lora(llm.model, self.sft_best_checkpoint_path)
                    save_lora(llm.model, self.best_checkpoint_path)
                    logger.info(
                        "SFT WARMUP EVAL (Step %d): Updated best metric to %s",
                        step_idx,
                        last_metric,
                    )

                eval_already = True

            # Get next SFT batch (cycle on epoch boundary)
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            # SFT training step (cross-entropy)
            llm.model.train()
            if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
                self._check_model_inputs(batch, hf_tokenizer, cfg)

            loss_metrics = self._compute_sft_loss(llm.model, batch)
            loss = loss_metrics["loss"] / grad_accum_step
            loss.backward()

            batch_idx += 1
            if batch_idx % grad_accum_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_idx += 1
                eval_already = False
                save_already = False
                pbar.update(1)

            # Log loss metrics
            sft_loss_metrics = {
                f"sft_warmup/{k}": get_item(v)
                for k, v in loss_metrics.items()
                if hasattr(v, "numel") and v.numel() == 1
            }
            metrics.update(sft_loss_metrics)
            pbar.set_postfix(**sft_loss_metrics)

            try:
                metrics["time/total"] = time.time() - t_start
                self.ml_logger.log_metrics(metrics)
            except Exception as e:
                rich_print(f"[red]Error logging metrics: {type(e).__name__}: {e}[/red]")
            torch.cuda.empty_cache()

        pbar.close()
        # Save end-of-warmup checkpoint
        save_lora(llm.model, f"{self.checkpoint_path}/sft_warmup_final")
        logger.info(
            "SFT warmup complete: %d steps, best_ppl=%.4f at step %d",
            num_steps,
            self.sft_best_ppl,
            self.sft_best_ppl_step,
        )

    def train(
        self,
        llm: HuggingFaceLLM | None = None,
        optimizer: Optimizer | Any | None = None,
        cfg: DictConfig | None = None,
        env: Environment | None = None,
        eval_env: Environment | None = None,
        eval_every: int | None = None,
        num_steps: int | None = None,
        num_substeps: int | None = None,
        checkpoint_name: str | None = None,
        name_or_identifier: str | None = None,
        **kwargs: Any,
    ) -> HuggingFaceLLM:
        """Two-phase training: SFT warmup then RL."""
        # Phase 1: SFT warmup
        if self.sft_warmup_steps > 0:
            logger.info("=== Phase 1: SFT Warmup (%d steps) ===", self.sft_warmup_steps)
            self.sft_warmup(
                llm=llm,
                optimizer=optimizer,
                cfg=cfg,
                env=env,
                eval_env=eval_env,
                checkpoint_name=checkpoint_name,
                name_or_identifier=name_or_identifier,
            )

        # Phase 2: RL training
        logger.info("=== Phase 2: RL Training ===")
        return super().train(
            llm=llm,
            optimizer=optimizer,
            cfg=cfg,
            env=env,
            eval_env=eval_env,
            eval_every=eval_every,
            num_steps=num_steps,
            num_substeps=num_substeps,
            checkpoint_name=checkpoint_name,
            name_or_identifier=name_or_identifier,
            **kwargs,
        )
