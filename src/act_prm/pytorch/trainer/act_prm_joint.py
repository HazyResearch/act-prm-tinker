"""
PyTorch Joint Trainer for Act-PRM: trains a single LLM to both
(1) generate thoughts from observed actions (gen_think), and
(2) generate standard thought-action rollouts (think_act).

Ported from act_prm/trainer/act_prm_joint.py (Tinker version).
"""

import logging
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


class ActPrmJointTrainer(RLTrainer):
    """
    PyTorch trainer for joint Act-PRM training.

    For each batch:
    1. Generate rollouts via gen_think (action-prompted thought generation) on act_prm env
    2. Train on combined trajectories: think_act_policy + policy (shuffled)

    Evaluates both gen_think (on act_prm env eval) and think_act (on eval_env).
    """

    def __init__(
        self,
        cfg: DictConfig,
        generator_cfg: DictConfig,
        checkpoint_path: str | None = None,
        log_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            cfg=cfg,
            generator_cfg=generator_cfg,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            **kwargs,
        )

        # Two generator constructors
        # think_act: standard rollouts (state -> thought, action)
        self.think_act_generator_constructor = self.get_generator_constructor(
            name="default",
            verbose=generator_cfg.verbose,
        )
        # gen_think: action-prompted thought generation (state, action -> thought)
        self.gen_think_generator_constructor = self.get_generator_constructor(
            **generator_cfg,
        )

        # Separate best metrics for each mode
        self.best_metric_think_act = -float("inf")
        self.best_metric_gen_think = -float("inf")

        self.best_think_act_checkpoint_path = join(
            self.checkpoint_path, "step_best_think_act"
        )
        self.best_gen_think_checkpoint_path = join(
            self.checkpoint_path, "step_best_gen_think"
        )

        import os
        os.makedirs(self.best_think_act_checkpoint_path, exist_ok=True)
        os.makedirs(self.best_gen_think_checkpoint_path, exist_ok=True)

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
        **generate_and_save_trajectories_kwargs: Any,
    ) -> HuggingFaceLLM:
        """
        Joint training loop: gen_think rollouts + think_act training.
        """
        llm = llm or self.llm
        optimizer = optimizer or self.optimizer
        cfg = cfg or self.cfg
        env = env or self.env
        eval_env = eval_env or self.eval_env
        hf_tokenizer = self.hf_tokenizer
        eval_every = eval_every or cfg.eval_every

        num_steps = num_steps or cfg.get("num_steps", None) or cfg.num_batches
        num_substeps = num_substeps or cfg.num_substeps
        dataloader_batch_size = 1

        wen_shuffle = len(env.datasets["train"])

        for batch_idx in range(0, num_steps):
            metrics = {
                "progress/batch": batch_idx,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (batch_idx + 1) / num_steps,
            }
            t_start = time.time()

            # === Evaluations ===
            if (
                eval_every > 0 and batch_idx % eval_every == 0
            ) or batch_idx == num_steps - 1:
                with timed("run_evals", metrics):
                    llm.model.eval()

                    # 1. Evaluate gen_think on act_prm env (eval split)
                    env.split = "eval"
                    gen_think_eval_metrics, _ = run_rollouts(
                        llm=llm,
                        hf_tokenizer=hf_tokenizer,
                        generator_constructor=self.gen_think_generator_constructor,
                        env=env,
                        cfg=cfg,
                        batch_id=batch_idx,
                        checkpoint_name="aprm_gen_think",
                        split="eval",
                        num_tries=cfg.eval_num_tries,
                        start_idx=0,
                        tasks_per_update=len(env),
                        name_or_identifier=(
                            f"Eval gen_think, Step {batch_idx}/{num_steps - 1}"
                        ),
                    )
                    metrics.update(gen_think_eval_metrics)
                    display_metrics(
                        gen_think_eval_metrics,
                        title=f"Gen-Think Eval, Step {batch_idx}",
                        style="bright_yellow",
                    )

                    # Check best gen_think metric
                    gen_think_metric_key = [
                        k
                        for k in gen_think_eval_metrics
                        if self.best_metric_name in k
                    ]
                    if gen_think_metric_key:
                        last_gen_think = gen_think_eval_metrics[gen_think_metric_key[0]]
                        if is_better(
                            last_gen_think,
                            self.best_metric_gen_think,
                            self.best_metric_name,
                        ):
                            self.best_metric_gen_think = last_gen_think
                            save_lora(llm.model, self.best_gen_think_checkpoint_path)
                            logger.info(
                                "Gen-think best %s: %f at step %d",
                                self.best_metric_name,
                                last_gen_think,
                                batch_idx,
                            )
                            metrics["gen_think/best_metric"] = last_gen_think
                            metrics["gen_think/best_step"] = batch_idx

                    # 2. Evaluate think_act on eval_env (train + eval splits)
                    for _split in ["train", "eval"]:
                        eval_env.split = _split
                        think_act_eval_metrics, _ = run_rollouts(
                            llm=llm,
                            hf_tokenizer=hf_tokenizer,
                            generator_constructor=self.think_act_generator_constructor,
                            env=eval_env,
                            cfg=cfg,
                            batch_id=batch_idx,
                            checkpoint_name="aprm_think_act",
                            split=_split,
                            num_tries=cfg.eval_num_tries,
                            start_idx=0,
                            tasks_per_update=len(eval_env),
                            name_or_identifier=(
                                f"Eval think_act ({_split}), "
                                f"Step {batch_idx}/{num_steps - 1}"
                            ),
                        )
                        metrics.update(think_act_eval_metrics)
                        display_metrics(
                            think_act_eval_metrics,
                            title=f"Think-Act {_split.upper()} Eval, Step {batch_idx}",
                            style="bright_cyan",
                        )

                        # Save best think_act on eval split
                        if _split == "eval":
                            think_act_metric_key = [
                                k
                                for k in think_act_eval_metrics
                                if self.best_metric_name in k
                            ]
                            if think_act_metric_key:
                                last_think_act = think_act_eval_metrics[
                                    think_act_metric_key[0]
                                ]
                                if is_better(
                                    last_think_act,
                                    self.best_metric_think_act,
                                    self.best_metric_name,
                                ):
                                    self.best_metric_think_act = last_think_act
                                    save_lora(
                                        llm.model,
                                        self.best_think_act_checkpoint_path,
                                    )
                                    logger.info(
                                        "Think-act best %s: %f at step %d",
                                        self.best_metric_name,
                                        last_think_act,
                                        batch_idx,
                                    )
                                    metrics["think_act/best_metric"] = last_think_act
                                    metrics["think_act/best_step"] = batch_idx

                                    try:
                                        self.replay_buffer.save_to_hf_dataset(
                                            self.best_replay_buffer_path
                                        )
                                    except SchemaInferenceError:
                                        logger.warning(
                                            "Failed to save replay buffer (empty?)"
                                        )

            # === Generate rollouts via gen_think ===
            env.split = "train"
            rl_start_idx = batch_idx * cfg.batch_size
            if rl_start_idx + cfg.batch_size > wen_shuffle:
                random.shuffle(env.datasets["train"])
                wen_shuffle += len(env.datasets["train"])

            train_rollout_metrics, new_trajectories = run_rollouts(
                llm=llm,
                hf_tokenizer=hf_tokenizer,
                generator_constructor=self.gen_think_generator_constructor,
                env=env,
                cfg=cfg,
                batch_id=batch_idx,
                checkpoint_name=checkpoint_name or "aprm_gen_think",
                split="train",
                num_tries=cfg.num_tries,
                start_idx=rl_start_idx,
                tasks_per_update=cfg.batch_size,
                name_or_identifier=(
                    f"Gen-think rollouts, Step {batch_idx}/{num_steps - 1}"
                ),
            )
            metrics.update(train_rollout_metrics)
            display_metrics(
                train_rollout_metrics,
                title=f"Gen-Think Train Rollouts, Step {batch_idx}",
                style="bright_cyan",
            )

            # Save think_act_policy trajectories to replay buffer
            for trajectory in new_trajectories["think_act_policy"]:
                self.replay_buffer.add_trajectory(trajectory)
            self.replay_buffer.save_to_hf_dataset(self.last_replay_buffer_path)

            # === Train on combined trajectories ===
            llm.model.train()

            all_trajectories = (
                new_trajectories["think_act_policy"] + new_trajectories["policy"]
            )
            random.shuffle(all_trajectories)

            train_loader, _minibatch_metrics = self.prepare_minibatch(
                new_trajectories=all_trajectories,
                hf_tokenizer=hf_tokenizer,
                batch_size=dataloader_batch_size,
                shuffle=True,
            )
            metrics.update(_minibatch_metrics)

            gradient_accumulation_steps = max(1, len(train_loader) // num_substeps)
            pbar_substep = tqdm(
                total=num_substeps,
                desc="Number of substeps",
                colour="blue",
                position=2,
            )
            pbar_dataloader = tqdm(
                train_loader,
                desc="Dataloader batches",
                colour="cyan",
                position=3,
            )

            for mini_batch_idx, mini_batch in enumerate(pbar_dataloader):
                if mini_batch_idx == 0 or (mini_batch_idx + 1) % 10 == 0:
                    self._check_model_inputs(mini_batch, hf_tokenizer, cfg)

                loss_metrics = self.compute_loss(
                    llm.model, mini_batch, fp32_loss=self.fp32_loss
                )
                loss = loss_metrics["loss"]
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (mini_batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar_substep.update(1)

                loss_metrics = {
                    f"train/{k}": get_item(v) for k, v in loss_metrics.items()
                }
                metrics.update(loss_metrics)
                pbar_dataloader.set_postfix(**loss_metrics)

            # Log metrics
            try:
                metrics["time/total"] = time.time() - t_start
                self.ml_logger.log_metrics(metrics)
            except Exception as e:
                logger.error("Error logging metrics: %s: %s", type(e).__name__, e)
            torch.cuda.empty_cache()

        # Load best think_act checkpoint at end
        llm.model = load_lora(llm.model, self.best_think_act_checkpoint_path)
        return llm
