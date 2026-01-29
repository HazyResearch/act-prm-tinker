"""
PyTorch RL Trainer for Hugging Face Transformers models
"""

import logging
import os
import random
import time
from os.path import join
from typing import Any

from omegaconf import DictConfig
from tqdm import tqdm

import torch
from torch.optim import Optimizer

from datasets.arrow_writer import SchemaInferenceError

from act_prm.lora import load_lora, save_lora
from act_prm.llm_handlers import HuggingFaceLLM
from act_prm.environments import Environment

from act_prm.trainer.tinker.utils import timed

from ..train import run_rollouts
from .base import display_metrics
from .rl import get_item, is_better, RLTrainer


logger = logging.getLogger(__name__)


class ActPrmForSftTrainer(RLTrainer):
    """
    PyTorch trainer for Act-PRM SFT trace generation with Hugging Face Transformers models
    """
    def __init__(
        self,
        cfg: DictConfig,
        generator_cfg: DictConfig,
        checkpoint_path: str | None = None,
        log_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cfg=cfg, generator_cfg=generator_cfg, checkpoint_path=checkpoint_path, log_path=log_path, **kwargs)

        # Checkpointing and best metrics
        self.best_checkpoint_path = join(self.checkpoint_path, "step_best")
        if not os.path.exists(self.best_checkpoint_path):
            os.makedirs(self.best_checkpoint_path)
        self.best_replay_buffer_path = join(self.checkpoint_path, "replay_buffer_best")
        self.last_replay_buffer_path = join(self.checkpoint_path, "replay_buffer")

        self.best_metric = float("-inf")
        self.best_metric_step = -1
        self.best_metric_name = cfg.best_metric

        # Redefine generator constructor for RL
        self.rl_generator_constructor = self.get_generator_constructor(
            name="default",
            verbose=generator_cfg.verbose,
        )
        self.action_prompt_generator_constructor = self.get_generator_constructor(
            **generator_cfg,
        )

    def train(
        self,
        llm: HuggingFaceLLM | None = None,
        optimizer: Optimizer | Any | None = None,
        cfg: DictConfig | None = None,
        env: Environment | None = None,
        eval_env: Environment | None = None,
        eval_every: int | None = None,
        # Specify training duration
        num_steps: int | None = None,
        num_substeps: int | None = None,
        # Other identifiers
        checkpoint_name: str | None = None,
        name_or_identifier: str | None = None,
        **generate_and_save_trajectories_kwargs: Any,
    ) -> HuggingFaceLLM:
        """
        Implement entire Policy Gradient training loop for Hugging Face Transformer (PEFT) model (llm.model)
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

        # 1. Determine mechanical dataset batch size and number of epochs
        num_steps = num_steps or cfg.get("num_steps", None) or cfg.num_batches
        num_substeps = num_substeps or cfg.num_substeps  # number of effective gradient updates per sampling batch
        dataloader_batch_size = 1 if cfg.get("group_size", 1) == 1 else 2  # HF behavior w/ batches and padding, also GPU poor
        # mini_batch_size = mini_batch_size or cfg.mini_batch_size
        # grad_accum_step = gradient_accumulation_steps or cfg.gradient_accumulation_steps or 1
        # dataloader_batch_size = mini_batch_size // grad_accum_step
        # dataloader_batch_size = max(2, dataloader_batch_size)  # HF behavior w/ batches and padding

        wen_shuffle = len(env.datasets["train"])

        for batch_idx in range(0, num_steps):
            metrics = {
                "progress/batch": batch_idx,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (batch_idx + 1) / num_steps,
            }
            t_start = time.time()

            # Run evaluations
            if (eval_every > 0 and batch_idx % eval_every == 0) or batch_idx == num_steps - 1:
                with timed("run_evals", metrics):
                    eval_env.split = "eval"
                    llm.model.eval()
                    eval_rollout_metrics, _ = run_rollouts(
                        llm=llm,
                        hf_tokenizer=hf_tokenizer,
                        generator_constructor=self.action_prompt_generator_constructor,
                        env=eval_env,
                        cfg=cfg,
                        batch_id=batch_idx,
                        checkpoint_name=checkpoint_name,
                        split="eval",
                        num_tries=cfg.eval_num_tries,
                        # Just use all eval tasks
                        start_idx=0,  
                        tasks_per_update=len(eval_env),
                        name_or_identifier=name_or_identifier,
                    )
                    metrics.update(eval_rollout_metrics)
                    display_metrics(eval_rollout_metrics, title=f"Rollout Eval Metrics, Step {batch_idx}", style="bright_yellow")

                # Save best checkpoints
                best_metric_key = [k for k in eval_rollout_metrics.keys() if self.best_metric_name in k][0]
                last_metric = eval_rollout_metrics[best_metric_key]
                if is_better(last_metric, self.best_metric, self.best_metric_name):
                    self.best_metric = last_metric
                    self.best_metric_step = batch_idx
                    save_lora(llm.model, self.best_checkpoint_path)
                    logger.info(
                        f"RL EVAL (Step {batch_idx}): "
                        f"Updated best metric to {last_metric} at step {batch_idx}"
                    )
                    metrics.update({
                        f"eval/{self.best_metric_name}": last_metric,
                        f"eval/{self.best_metric_name}_best": self.best_metric,
                        f"eval/{self.best_metric_name}_best_step": self.best_metric_step,
                    })
                    try:  # Saving replay buffer
                        self.replay_buffer.save_to_hf_dataset(self.best_replay_buffer_path)
                        logger.info("Saved best replay buffer to %s", self.best_replay_buffer_path)
                    except SchemaInferenceError:
                        logger.warning(
                            "Failed to save best replay buffer to %s\nIs replay buffer empty?",
                            self.best_replay_buffer_path
                        )

            # Generate and save trajectories to a HF Dataset
            _save_rollouts_every = cfg.get("save_rollouts_every", num_steps)
            do_save_rollouts = (batch_idx + 1) % _save_rollouts_every == 0 or (batch_idx + 1 == num_steps)
            if _save_rollouts_every > 0 and do_save_rollouts:
                self.generate_and_save_trajectories(
                    save_generator_constructor=self.action_prompt_generator_constructor,
                    save_batch_idx=batch_idx,
                    llm=llm,
                    save_env=env,
                    cfg=cfg,
                    hf_tokenizer=hf_tokenizer,
                    save_name_or_identifier="Stage 1: SFT Generation with Act-PRM LLM",
                    trajectory_key="think_act_policy",
                    dataset_prefix="mzio/aprm-sft_genthinkact",
                    dataset_suffix="-ap1",
                )

            # 1. Sample rollouts for training
            env.split = "train"
            rl_start_idx = batch_idx * cfg.batch_size
            if rl_start_idx + cfg.batch_size > wen_shuffle:
                random.shuffle(env.datasets["train"])
                wen_shuffle += len(env.datasets["train"])
            
            train_rollout_metrics, new_trajectories = run_rollouts(
                llm=llm,
                hf_tokenizer=hf_tokenizer,
                generator_constructor=self.action_prompt_generator_constructor,
                env=env,
                cfg=cfg,
                batch_id=batch_idx,
                checkpoint_name=checkpoint_name,
                split="train",
                num_tries=cfg.num_tries,
                start_idx=rl_start_idx,
                tasks_per_update=cfg.batch_size,
                name_or_identifier=name_or_identifier,
            )
            metrics.update(train_rollout_metrics)
            display_metrics(train_rollout_metrics, title=f"Rollout Training Metrics, Step {batch_idx}", style="bright_cyan")
            
            # Save replay buffer samples
            for trajectory in new_trajectories["policy"]:
                self.replay_buffer.add_trajectory(trajectory)
            self.replay_buffer.save_to_hf_dataset(self.last_replay_buffer_path)

            # 2. Update policy LLM with generated rollouts
            llm.model.train()
            
            train_loader, _minibatch_metrics = self.prepare_minibatch(
                new_trajectories=new_trajectories["policy"],
                hf_tokenizer=hf_tokenizer,
                batch_size=dataloader_batch_size,
                shuffle=True,
            )
            metrics.update(_minibatch_metrics)  # empty {} for now

            # For now, auto-calculate gradient accumulation steps based on num_substeps
            gradient_accumulation_steps = max(1, len(train_loader) // num_substeps)
            pbar_substep = tqdm(total=num_substeps, desc="Number of substeps", colour="blue", position=2)
            pbar_dataloader = tqdm(train_loader, desc="Dataloader batches", colour="cyan", position=3)

            for mini_batch_idx, mini_batch in enumerate(pbar_dataloader):
                # Sanity-check model inputs
                if mini_batch_idx == 0 or (mini_batch_idx + 1) % 10 == 0:
                    self._check_model_inputs(mini_batch, hf_tokenizer, cfg)

                loss_metrics = self.compute_loss(llm.model, mini_batch, fp32_loss=self.fp32_loss)
                loss = loss_metrics["loss"]
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (mini_batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar_substep.update(1)

                loss_metrics = {
                    f"train/{k}": get_item(v) for k, v in loss_metrics.items()
                    if v.numel() == 1  # only keep scalar metrics
                }
                metrics.update(loss_metrics)
                pbar_dataloader.set_postfix(**loss_metrics)

            # Log metrics
            try:
                metrics["time/total"] = time.time() - t_start
                self.ml_logger.log_metrics(metrics)  # increments each time
            except Exception as e:
                _error_class = e.__class__.__name__
                _error_message = str(e)
                logger.error(f"Error logging metrics: {_error_class}: {_error_message}")
                for k, v in metrics.items():
                    logger.error(f"-> {k}: {type(v)}")
                breakpoint()
            torch.cuda.empty_cache()

        # Load best model checkpoint
        llm.model = load_lora(llm.model, self.best_lm_checkpoint_path)
        self.generate_and_save_trajectories(
            save_generator_constructor=self.action_prompt_generator_constructor,
            save_batch_idx=num_steps - 1,  # Last batch
            llm=llm,
            save_env=env,
            cfg=cfg,
            hf_tokenizer=hf_tokenizer,
            save_name_or_identifier="Stage 1: Final SFT Generation with Best Act-PRM LLM",
            trajectory_key="think_act_policy",
            dataset_prefix="mzio/aprm-sft_genthinkact",
            dataset_suffix="-ap1_best",
        )
        return llm
