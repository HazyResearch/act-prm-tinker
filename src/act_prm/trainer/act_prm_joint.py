"""
Tinker Trainers for fully synchronous Act-PRM training in a joint-RL setup

For each batch, we train the *same* model to:  
1. Generate thoughts, from action-conditioned prompts (state, action, <thought that lead to action>)
2. Generate thoughts and actions, from state prompts (state, <thought, action>)

We do N rounds of just training for 1., while training on the generated artifacts for both 1. and 2.,
before doing RL training for 2.

```bash
uv run python main.py \
--is_async \
--env_config act_prm/browsecomp_100_hide_obs \
--eval_env_config browsecomp_plus/search_hide_obs \
--generator_config aprm_qwen3_ap \
--trainer_config qwen3_4b_aprm_joint100 \
--replay_buffer_config default \
--log_path ./logs \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--seed 42 --replicate 0 --verbose
```
"""

import logging
import random
import time
from typing import Any, Callable

import torch
from omegaconf import DictConfig
from rich import print as rich_print
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import tinker
from tinker import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.checkpoint_utils import save_checkpoint_async
from tinker_cookbook.utils import ml_log

from datasets.arrow_writer import SchemaInferenceError
from transformers import PreTrainedTokenizerBase

from ..environments import Environment, EnvironmentState
from ..environments.act_prm import ActPrmEnv
from ..generator.tinker import TinkerGenerator
from ..llm_handlers.action_utils import get_actions
from ..replay_buffer import ReplayBuffer
from ..replay_buffer.types import StateActionSample, Trajectory

from .rl import RLTrainer, _save_trajectories_to_hf_dataset
from .tinker.utils import (
    gather_with_progress,
    save_checkpoint_and_get_sampling_client,
    split_list,
    timed,
)
from .train import is_better, run_rollouts

logger = logging.getLogger(__name__)
console = Console()


def display_metrics(
    metrics: dict[str, Any],
    title: str | None = None,
    style: str = "bright_yellow",
) -> None:
    # Display metrics
    table = Table(title=title, style=style)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="left")
    for k, v in metrics.items():
        table.add_row(k, str(v))
    console.print(table)


class ActPrmJointTrainer(RLTrainer):
    """
    Trainer for fully synchronous Act-PRM training
    """
    def __init__(
        self, 
        cfg: DictConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
        generator_cfg: DictConfig,
        replay_buffer: ReplayBuffer,
        env: ActPrmEnv,
        eval_env: Environment,
        ml_logger: ml_log.Logger,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            cfg,
            training_client,
            service_client,
            generator_cfg,
            replay_buffer,
            env,
            eval_env,
            ml_logger,
            hf_tokenizer,
            **kwargs,
        )
        # RL / Evaluation generator: does standard rollouts, see act_prm/generator/default.py
        self.think_act_generator_constructor = self.get_generator_constructor(
            name="default",
            verbose=generator_cfg.verbose,
        )
        self.gen_think_generator_constructor = self.get_generator_constructor(
            **generator_cfg,
        )
        # Training configs -> ignored currently
        # self.action_prompts = cfg.get("action_prompts", False)  # if False just do SFT
        # self.num_batches_action_prompts = cfg.get("num_batches_action_prompts", 0)
        # self.advantage_threshold = cfg.get("advantage_threshold", None)

        # Paths to the best sampling clients
        self.best_gen_think_sampling_client_path = ""
        self.best_gen_think_state_path = ""
        
        self.best_think_act_sampling_client_path = ""
        self.best_think_act_state_path = ""

        self.best_metric_think_act = -float("inf")
        self.best_metric_gen_think = -float("inf")

    # Modified from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L989
    async def train(
        self, 
        start_batch: int, 
        end_batch: int, 
        cfg: DictConfig | None = None,
        env: Environment | None = None,
        eval_env: Environment | None = None,
        generator_constructor: Callable[..., TinkerGenerator] | None = None,
        checkpoint_name: str | None = None,
    ) -> str:
        """
        Implement fully synchronous on-policy training with Tinker

        For each training batch:
        1. Loads the most recent checkpoint and determines how we generate rollouts
        2. Generates rollouts (optionally running on the evaluation environment)
        3. Performs a policy update
        """
        cfg = cfg or self.cfg
        env = env or self.env
        eval_env = eval_env or self.eval_env
        # Ignored -> We define this in class init
        # generator_constructor = generator_constructor or self.generator_constructor

        # Initial sampling client
        sampling_client, _ = await save_checkpoint_and_get_sampling_client(
            training_client=self.training_client,
            i_batch=start_batch,
            log_path=cfg.log_path,
            save_every=cfg.save_every,
            start_batch=start_batch,
        )

        model_name = cfg.model_name or self.training_client.get_info().model_data.model_name
        hf_tokenizer = self.hf_tokenizer or self.training_client.get_tokenizer()
        # ^Same as tinker_cookbook.tokenizer_utils.get_tokenizer(cfg.model_name)?
        renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(model_name)
        renderer = renderers.get_renderer(renderer_name, hf_tokenizer)
        logger.info("Using renderer: %s", renderer_name)

        async def do_eval(
            sampling_client: tinker.SamplingClient,
            generator_constructor,
            eval_env: Environment,
            batch_id: int,
            checkpoint_prefix: str,
            split: str,
            name_or_identifier: str,
            metrics: dict[str, Any],
            best_val_metric: float,
        ) -> tuple[dict[str, Any], float, bool, dict[str, Any] | None, list[Trajectory]]:
            """
            Convenience function for running evaluations with shared parameters

            Returns:
            - metrics: rollout metrics from run_rollouts()
            - last_val_metric: last metric value from run_rollouts()
            - new_best_metric: True if last_val_metric is better than best_val_metric
            - best_metric_path_dict: paths to best Tinker sampling client and state
            - new_trajectories: new trajectories from run_rollouts()
            """
            num_eval_tasks = len(eval_env.datasets["eval"])
            new_best_metric = False  # set to True if eval_metric is better
            best_metric_path_dict: dict[str, Any] | None = None
            
            eval_rollout_metrics, new_trajectories = await run_rollouts(
                sampling_client=sampling_client,
                renderer=renderer,
                hf_tokenizer=hf_tokenizer,
                generator_constructor=generator_constructor,
                env=eval_env,
                cfg=cfg,
                batch_id=batch_id,
                checkpoint_name=checkpoint_prefix,
                split=split,
                num_tries=cfg.eval_num_tries,
                start_idx=0,
                tasks_per_update=num_eval_tasks,
                name_or_identifier=name_or_identifier,
            )
            metrics.update(eval_rollout_metrics)
            display_metrics(
                eval_rollout_metrics,
                title=f"{split.upper()} Split Rollouts, Train Step {batch_id}/{num_batches-1}",
                style="bright_yellow" if split == "eval" else "bright_cyan",
            )
            # Get last metric to validate against best_metric
            _metric_prefix = f"{checkpoint_prefix}_{split}"
            best_metric_key = f"{_metric_prefix}/try_{cfg.eval_num_tries-1}/{cfg.best_metric}"
            last_val_metric = eval_rollout_metrics[best_metric_key]
            best_ckpt_name = f"{_metric_prefix}_{batch_id:04d}_best"
            if (
                is_better(last_val_metric, best_val_metric, cfg.best_metric)
                and split == "eval"
            ):  # Only save best checkpoints on eval splits
                new_best_metric = True
                path_dict = await save_checkpoint_async(
                    training_client=self.training_client,
                    name=best_ckpt_name,
                    log_path=cfg.log_path,
                    loop_state={"batch": batch_id},
                    kind="both",
                )
                logger.info(
                    "Saved best %s sampling_client to %s",
                    checkpoint_prefix, path_dict["sampler_path"],
                )
                logger.info(
                    "Saved best %s state to %s", checkpoint_prefix, path_dict["state_path"]
                )
                logger.info(
                    "Updated best %s %s to %f at batch %d",
                    checkpoint_prefix, cfg.best_metric, last_val_metric, batch_id,
                )
                metrics.update({
                    f"{checkpoint_prefix}/best_batch": batch_id,
                    f"{checkpoint_prefix}/best_metric": last_val_metric,
                    f"{checkpoint_prefix}/best_sampling_client_path": path_dict["sampler_path"],
                    f"{checkpoint_prefix}/best_state_path": path_dict["state_path"],
                })
                best_metric_path_dict = path_dict
            return (
                metrics, last_val_metric, new_best_metric, best_metric_path_dict, new_trajectories
            )
        
        num_batches = end_batch - start_batch
        wen_shuffle = len(env.datasets["train"])
        
        pbar = tqdm(
            range(start_batch, end_batch),
            desc="Training to (think from actions) and (act from thoughts)",
            colour="blue",
            position=1,
        )
        for batch_idx, overall_batch_idx in enumerate(pbar):
            metrics = {
                "progress/batch": batch_idx,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (batch_idx + 1) / num_batches,
            }
            t_start = time.time()
            
            # Run evaluations
            if (
                cfg.eval_every > 0 
                and ((batch_idx + 1) % cfg.eval_every == 0 or batch_idx + 1 == cfg.num_batches)
            ):
                with timed("run_evals", metrics):
                    # Run `gen_think` evaluation on held-out eval tasks
                    env.split = "eval"
                    eval_results = await do_eval(
                        sampling_client=sampling_client,
                        generator_constructor=self.gen_think_generator_constructor,
                        eval_env=env,
                        batch_id=batch_idx,
                        checkpoint_prefix="aprm_gen_think",
                        split="eval",
                        name_or_identifier=(
                            f"Evaluating action-prompted thought generation (gen_think), "
                            f"Train Step {batch_idx} / {num_batches - 1} on eval split"
                        ),
                        metrics=metrics,
                        best_val_metric=self.best_metric_gen_think,
                    )
                    metrics, last_val_metric, new_best_metric, best_metric_path_dict, _ = eval_results
                    if new_best_metric:
                        self.best_metric_gen_think = last_val_metric
                        self.best_gen_think_sampling_client_path = best_metric_path_dict["sampler_path"]
                        self.best_gen_think_state_path = best_metric_path_dict["state_path"]
                        
                    # Run `think_act` evaluation on both train and eval splits
                    for _split in ["train", "eval"]:
                        eval_env.split = _split
                        _name_or_identifier = (
                            f"Evaluating standard rollouts (think_act), "
                            f"Train Step {batch_idx} / {num_batches - 1} on eval split"
                        )
                        eval_results = await do_eval(
                            sampling_client=sampling_client,
                            generator_constructor=self.think_act_generator_constructor,
                            eval_env=eval_env,
                            batch_id=batch_idx,
                            checkpoint_prefix="aprm_think_act",
                            split=_split,
                            name_or_identifier=_name_or_identifier,
                            metrics=metrics,
                            best_val_metric=self.best_metric_think_act,
                        )
                        metrics, last_val_metric, new_best_metric, best_metric_path_dict, _ = eval_results
                        if new_best_metric:
                            self.best_metric_think_act = last_val_metric
                            self.best_think_act_sampling_client_path = best_metric_path_dict["sampler_path"]
                            self.best_think_act_state_path = best_metric_path_dict["state_path"]
                            try:  # Saving replay buffer
                                self.replay_buffer.save_to_hf_dataset(self.best_replay_buffer_path)
                                logger.info("Saved best replay buffer to %s", self.best_replay_buffer_path)
                            except SchemaInferenceError:
                                logger.warning(
                                    "Failed to save best replay buffer to %s\nIs replay buffer empty?",
                                    self.best_replay_buffer_path
                                )
            
            # 1. Sample rollouts for training
            env.split = "train"
            rl_start_idx = batch_idx * cfg.batch_size
            if rl_start_idx + cfg.batch_size > wen_shuffle:
                random.shuffle(env.datasets["train"])
                wen_shuffle += len(env.datasets["train"])
            _ckeckpoint_name = (
                f"{checkpoint_name}-aprm_gen_think"
                if checkpoint_name is not None
                else "aprm_gen_think"
            )
            train_rollout_metrics, new_trajectories = await run_rollouts(
                sampling_client=sampling_client,
                renderer=renderer,
                hf_tokenizer=hf_tokenizer,
                generator_constructor=self.gen_think_generator_constructor,
                env=env,
                cfg=cfg,
                batch_id=batch_idx,
                checkpoint_name=_checkpoint_name,
                split="train",
                num_tries=cfg.num_tries,
                start_idx=rl_start_idx,
                tasks_per_update=cfg.batch_size,
                name_or_identifier=(
                    f"Generating rollouts via action-prompts (gen_think),"
                    f"Train Step {batch_idx} / {num_batches - 1}"
                ),
            )
            metrics.update(train_rollout_metrics)

            # Save replay buffer samples
            for trajectory in new_trajectories["think_act_policy"]:
                self.replay_buffer.add_trajectory(trajectory)
            self.replay_buffer.save_to_hf_dataset(self.last_replay_buffer_path)

            # 2. Update policy LLM with combined rollouts
            all_trajectories = new_trajectories["think_act_policy"] + new_trajectories["policy"]
            random.shuffle(all_trajectories)
            data_D, prepare_minibatch_metrics = await self.prepare_minibatch(
                new_trajectories=all_trajectories,
                service_client=self.service_client,
                model_name=cfg.model_name,
                kl_penalty_coef=cfg.kl_penalty_coef,
                kl_discount_factor=cfg.kl_discount_factor,
            )
            sampling_client, update_metrics = await self.do_train_step_and_get_sampling_client(
                batch_idx=batch_idx,
                training_client=self.training_client,
                data_D=data_D,
                prepare_minibatch_metrics=prepare_minibatch_metrics,
                loss_fn="importance_sampling",
                checkpoint_name=checkpoint_name,
            )

            # Log metrics
            metrics.update(update_metrics)
            metrics["time/total"] = time.time() - t_start
            self.ml_logger.log_metrics(metrics, step=overall_batch_idx)
            pbar.set_postfix(**{k.split("/")[-1]: v for k, v in metrics.items()})

        return self.best_think_act_sampling_client_path
