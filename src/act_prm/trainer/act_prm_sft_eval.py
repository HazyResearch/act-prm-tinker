"""
Tinker Trainers for fully synchronous Act-PRM training
"""

import logging
import random
import time
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig
from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

import tinker
from tinker import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.checkpoint_utils import save_checkpoint_async
from tinker_cookbook.utils import ml_log

from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from transformers import PreTrainedTokenizerBase

from ..environments import Environment
from ..environments.act_prm import ActPrmEnv
from ..generator.tinker import TinkerGenerator
from ..replay_buffer import ReplayBuffer
from ..replay_buffer.types import Trajectory

from .rl import RLTrainer
from .tinker.update import train_step
from .tinker.utils import save_checkpoint_and_get_sampling_client, split_list, timed
from .train import is_better, run_rollouts

logger = logging.getLogger(__name__)
console = Console()


class ActPrmSftEvalTrainer(RLTrainer):
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
        # Evaluation generator does standard rollouts, see act_prm/generator/default.py
        self.eval_generator_constructor = self.get_generator_constructor(
            name="default",
            verbose=generator_cfg.verbose,
        )
        self.action_prompt_generator_constructor = self.get_generator_constructor(
            # name="action_prompt_act_prm",
            # keep_top_k=cfg.get("action_prompt_keep_top_k", None),
            # **{k: v for k, v in generator_cfg.items() if k != "name"}
            **generator_cfg,
        )

    def _save_trajectories_to_hf_dataset(
        self,
        trajectories: list[Trajectory],
        dataset_name: str,
    ) -> None:
        """
        Save a list of trajectories to a HF Dataset
        """
        ds_samples = [
            {"state": step.state, "action": step.action, "tools": step.tools, "reward": step.reward}
            for trajectory in trajectories
            for step in trajectory.episode_steps
        ]
        Dataset.from_list(ds_samples).push_to_hub(
            dataset_name,
            private=False,
        )

    def _get_sft_datums_and_substeps(
        self,
        data_D: list[tinker.Datum],
        minibatch_size: int | None = None,
        num_epochs: int | None = None,
        num_substeps: int | None = None,
        num_substeps_per_epoch: int | None = None,
    ) -> tuple[list[tinker.Datum], int, int]:
        """
        Determines a training set from data_D, number of total updates, and minibatch size,
        from some combo of:
        - minibatch_size: number of datums per update
        - num_epochs: number of total loops of the data_D
        - num_substeps: number of total updates
        - num_substeps_per_epoch: number of updates per epoch
        """
        train_data_D: list[tinker.Datum] = []  # Build this as the entire set of datums to iterate through
        total_samples: int = 0  # to fill in
        assert num_epochs or num_substeps, "Either num_epochs or num_substeps must be specified"

        if num_epochs:
            total_samples = num_epochs * len(data_D)
            for _ in range(num_epochs):
                train_data_D.extend(random.sample(data_D, k=len(data_D)))

            if minibatch_size:
                num_substeps = total_samples // minibatch_size

            elif num_substeps:
                minibatch_size = total_samples // num_substeps

            elif num_substeps_per_epoch:
                num_substeps = num_substeps_per_epoch * num_epochs
                minibatch_size = total_samples // num_substeps

        elif num_substeps:
            if minibatch_size:
                total_samples = num_substeps * minibatch_size

            elif num_substeps_per_epoch:
                minibatch_size = len(train_data_D) // num_substeps_per_epoch
                total_samples = num_substeps * minibatch_size

            # Sample from data_D to hit total_samples
            if total_samples < len(data_D):
                train_data_D = random.sample(data_D, k=total_samples)
            else:
                while len(train_data_D) < total_samples:
                    train_data_D.extend(random.sample(data_D, k=len(data_D)))

                if len(train_data_D) >= total_samples:
                    train_data_D = train_data_D[:total_samples]
                else:  # Fill in the rest
                    train_data_D.extend(random.sample(data_D, k=total_samples - len(train_data_D)))
        else:
            raise ValueError("Either num_epochs or num_substeps must be specified")

        return train_data_D, num_substeps, minibatch_size

    async def do_sft_and_eval_loop(
        self,
        training_client: tinker.TrainingClient,
        data_D: list[tinker.Datum],
        eval_env: Environment,
        cfg: DictConfig | None = None,
        loop_id: int = 0,
        mini_batch_size: int | None = None,
        num_epochs: int | None = None,
        num_substeps: int | None = None,
        num_substeps_per_epoch: int | None = None,
        inner_eval_every: int = 10,
        inner_eval_num_tries: int = 1,
        eval_best_metric: str = "accuracy",
        metrics: dict[str, Any] | None = None,
        **run_rollouts_kwargs: Any,
    ) -> tuple[dict[str, list[Any]], dict[str, Any]]:
        """
        SFT a new policy LLM with the thought-action rollouts, periodically evaluating the LLM
        """
        cfg = cfg or self.cfg
        metrics = metrics or {}
        all_eval_metrics: dict[str, list[Any]] = {
            "batch_id": [],
        }
        eval_rollout_metrics: dict[str, Any] = {}
        # Use for sampling-based evaluation
        sampling_client: tinker.SamplingClient | None = None

        # Get training data
        train_data_D, num_substeps, _ = self._get_sft_datums_and_substeps(
            data_D,
            mini_batch_size,
            num_epochs,
            num_substeps,
            num_substeps_per_epoch,
        )
        logger.info("Got SFT dataset of size %d from %d datums", len(train_data_D), len(data_D))
        # breakpoint()

        # Split into batches and train
        batches = split_list(train_data_D, min(num_substeps, len(train_data_D)))

        logger.info("Split SFT dataset into %d batches", len(batches))
        # breakpoint()
        pbar = tqdm(
            enumerate(batches), 
            desc="Training with SFT (do_sft_and_eval_loop)", 
            leave=False, 
            colour="blue",
        )
        eval_idx = 0
        for batch_id, batch_data_D in pbar:
            if (
                (batch_id % inner_eval_every == 0 and sampling_client is not None)
                or (batch_id == len(batches) - 1)
            ):
                # Evaluate the LLM
                with timed("run_evals_with_sft", metrics):
                    eval_env.split = "eval"
                    eval_rollout_metrics, _ = await run_rollouts(
                        sampling_client=sampling_client,
                        generator_constructor=self.eval_generator_constructor,
                        env=eval_env,
                        cfg=cfg,
                        batch_id=batch_id,
                        checkpoint_name=f"sft_eval_{loop_id:03d}",  # prefix for metric keys
                        split="eval",
                        num_tries=inner_eval_num_tries,
                        start_idx=0,
                        tasks_per_update=len(eval_env),
                        **run_rollouts_kwargs,
                    )
                    # all_eval_metrics.append(eval_rollout_metrics)
                    # metrics.update(eval_rollout_metrics)
                    all_eval_metrics["batch_id"].append(batch_id)
                    for k, v in eval_rollout_metrics.items():
                        if k not in all_eval_metrics:
                            all_eval_metrics[k] = []
                        all_eval_metrics[k].append(v)
                    # Display metrics
                    table = Table(
                        title=f"SFT Eval Loop {loop_id}, Eval {eval_idx}, Batch {batch_id}",
                        style="bright_yellow",
                    )
                    table.add_column("Metric", justify="left")
                    table.add_column("Last Value", justify="left")
                    table.add_column("All Values", justify="left")
                    for k, v in eval_rollout_metrics.items():
                        table.add_row(k, str(v), str(all_eval_metrics[k]))
                    console.print(table)
                    eval_idx += 1

            # Do one SFT update
            with timed("sft_update", metrics):
                _ = await train_step(
                    batch_data_D,
                    training_client,
                    cfg.learning_rate,
                    num_substeps=1,
                    loss_fn="cross_entropy",
                )
            path_dict = await save_checkpoint_async(
                training_client=training_client,
                name=f"sft_eval_{batch_id:06d}",
                log_path=cfg.log_path,
                loop_state={"batch": batch_id},
                kind="both",
            )
            sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
            logger.info("Created new sampling client for SFT evaluation at path %s", path_dict["sampler_path"])
            # breakpoint()
            pbar.set_postfix(**{k.split("/")[-1]: v for k, v in eval_rollout_metrics.items()})
        
        # Get best evaluation metrics (over eval_idx)
        best_metric_keys = [k for k in all_eval_metrics.keys() if eval_best_metric in k]
        assert len(best_metric_keys) > 0, (
            f"Best metric key {eval_best_metric} not found in {all_eval_metrics.keys()}"
        )
        best_eval_idx = np.argmax([all_eval_metrics[k] for k in best_metric_keys]).item()
        best_eval_metrics = {}
        for k, v in all_eval_metrics.items():
            k = k.replace(f"sft_eval_{loop_id:03d}", "sft_eval_max")  # counts across eval_idx
            best_eval_metrics[k] = v[best_eval_idx]
        # Keep track of best_eval_idx
        best_eval_metrics["sft_eval_max/best_eval_idx"] = best_eval_idx
        return all_eval_metrics, best_eval_metrics

    async def do_eval_loop(
        self,
        sampling_client: tinker.SamplingClient,
        batch_id: int,
        loop_id: int,
        env: Environment,
        eval_env: Environment,
        cfg: DictConfig | None = None,
        num_tasks: int | None = None,
        renderer: renderers.Renderer | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> tuple[dict[str, list[Any]], dict[str, Any]]:
        """
        Evaluate the sampling client by generating a new dataset of thoguht-action trajectories,
        and training another LLM via SFT on this data
        """
        cfg = cfg or self.cfg

        # Generate thought-action rollouts
        num_tasks = num_tasks or len(env)
        env.split = "train"
        logger.info("Generating thought-action rollouts for %d tasks", num_tasks)
        _, new_trajectories = await run_rollouts(
            sampling_client=sampling_client,
            renderer=renderer,
            hf_tokenizer=hf_tokenizer,
            generator_constructor=self.action_prompt_generator_constructor,
            env=env,
            cfg=cfg,
            batch_id=batch_id,
            checkpoint_name=f"sft_gen_{batch_id:06d}",
            split="train",
            num_tries=1,
            start_idx=0,
            tasks_per_update=num_tasks,
        )
        all_new_trajectories = new_trajectories["thought_action_policy"]

        # Create a new HF dataset from the thought-action rollouts
        _ds_name = []
        for delim in ["-enco=", "-geco=", "-se=", "-re="]:
            _ds_name.append(self.run_name.split(delim)[-1].split("-")[0])
        _ds_name = "_".join(_ds_name)
        try:
            self._save_trajectories_to_hf_dataset(
                trajectories=all_new_trajectories,
                dataset_name=f"mzio/aprm_sft-{_ds_name}-{batch_id:04d}",  # hardcoded hack for now
            )
            logger.info("Saved thought-action rollouts to HF Dataset: %s", f"mzio/aprm_sft-{_ds_name}-{batch_id:04d}")
        except Exception as e:
            logger.error("Failed to save thought-action rollouts to HF Dataset: %s", e)
        
        # Create Tinker datums
        data_D, _ = await self.prepare_sft_minibatch(
            new_trajectories=all_new_trajectories,
        )
        logger.info("Create SFT dataset of size %d from %d thought-action rollouts", len(data_D), len(all_new_trajectories))

        # Initialize a new policy LLM for evaluating the Act-PRM model's generation quality
        sft_training_client = await self.service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )
        logger.info("Created new SFT training client")

        # Evaluate the current sampling_client LLM based on the data it generated
        all_eval_metrics, best_eval_metrics = await self.do_sft_and_eval_loop(
            training_client=sft_training_client,
            data_D=data_D,
            eval_env=eval_env,
            cfg=cfg,
            loop_id=loop_id,
            # for run_rollouts
            renderer=renderer,
            hf_tokenizer=hf_tokenizer,
            # for sft_and_eval_loop SFT training + eval metrics
            **cfg.sft_eval_kwargs,
        )
        return all_eval_metrics, best_eval_metrics

    async def prepare_sft_minibatch(
        self,
        new_trajectories: list[Trajectory],
    ) -> tuple[list[tinker.Datum], dict[str, Any]]:
        """
        Prepare a "minibatch" of trajectories for SFT training
        """
        metrics = {}
        # Assemble training data
        with timed("assemble_training_data", metrics):
            data_D: list[tinker.Datum] = []
            for trajectory in new_trajectories:
                for episode_step in trajectory.episode_steps:
                    sa_input_ids = episode_step.state_action_tokens
                    input_tokens = sa_input_ids[:-1]
                    target_tokens = sa_input_ids[1:]
                    target_state_len = episode_step.state_len - 1
                    target_action_len = len(target_tokens) - target_state_len
                    weight = getattr(episode_step, "advantage", 1.0)
                    weights = [0.0] * target_state_len + [weight] * target_action_len
                    data_D.append(
                        tinker.Datum(
                            model_input=tinker.ModelInput.from_ints(input_tokens),
                            loss_fn_inputs={
                                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                                "weights": TensorData.from_torch(torch.tensor(weights)),
                            },
                        )
                    )
        return data_D, metrics


    async def do_rl_loop(
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
        Run an RL training loop for a given generator (TinkerActPrmGenerator or TinkerActionPromptActPrmGenerator)
        """
        return await super().train(
            start_batch=start_batch,
            end_batch=end_batch,
            cfg=cfg,
            env=env,
            eval_env=eval_env,
            generator_constructor=generator_constructor,
            checkpoint_name=checkpoint_name,
        )

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
    ) -> None:
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
        generator_constructor = generator_constructor or self.generator_constructor

        # Paths to the best sampling clients
        best_sampling_client_path = ""

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
        
        # -------- Act-PRM training (generate thoughts from action-prompted states) --------
        logger.info("Starting Act-PRM model training")
        num_batches = end_batch - start_batch
        sft_eval_loop_id = 0
        for batch_idx in range(start_batch, end_batch):
            metrics = {
                "progress/batch": batch_idx,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (batch_idx + 1) / num_batches,
            }
            t_start = time.time()

            # Run action-likelihood evaluations
            if cfg.eval_every > 0 and batch_idx % cfg.eval_every == 0:
                with timed("run_evals_act_probs", metrics):
                    self.env.split = "eval"
                    eval_rollout_metrics, _ = await run_rollouts(
                        sampling_client=sampling_client,
                        renderer=renderer,
                        hf_tokenizer=hf_tokenizer,
                        generator_constructor=self.action_prompt_generator_constructor,
                        env=self.env,
                        cfg=cfg,
                        batch_id=batch_idx,
                        checkpoint_name=checkpoint_name,
                        split="eval",
                        num_tries=cfg.eval_num_tries,
                        # Just use all eval tasks
                        start_idx=0,  
                        tasks_per_update=len(self.env),
                    )
                    metrics.update(eval_rollout_metrics)

            # Run SFT'ing another LLM evaluations
            if cfg.sft_eval_every > 0 and batch_idx % cfg.sft_eval_every == 0 and batch_idx > 0:
                with timed("run_evals_act_probs", metrics):
                    sft_eval_metrics, best_sft_eval_metrics = await self.do_eval_loop(
                        sampling_client=sampling_client,
                        batch_id=batch_idx,
                        env=self.env,
                        eval_env=self.eval_env,
                        cfg=cfg,
                        loop_id=sft_eval_loop_id,
                        # for run_rollouts in do_sft_and_eval_loop
                        renderer=renderer,
                        hf_tokenizer=hf_tokenizer,
                    )
                    metrics.update(sft_eval_metrics)
                    metrics.update(best_sft_eval_metrics)

                # Save best ActPRM LLM checkpoints
                best_metric_keys = [
                    k for k in best_sft_eval_metrics.keys() if cfg.best_metric in k
                ]
                assert len(best_metric_keys) > 0, (
                    f"Best metric key {cfg.best_metric} not in {best_sft_eval_metrics.keys()}"
                )
                last_metric = best_sft_eval_metrics[best_metric_keys[0]]
                best_ckpt_name = (
                    f"{batch_idx:06d}_best"
                    if checkpoint_name is None
                    else f"{checkpoint_name}_{batch_idx:06d}_best"
                )
                if is_better(last_metric, self.best_metric, cfg.best_metric):
                    self.best_metric = last_metric
                    path_dict = await save_checkpoint_async(
                        training_client=self.training_client,
                        name=best_ckpt_name,
                        log_path=cfg.log_path,
                        loop_state={"batch": batch_idx},
                        kind="both",
                    )
                    best_sampling_client_path = path_dict["sampler_path"]
                    logger.info("Saved best sampling client to %s", best_sampling_client_path)
                    logger.info(
                        "Updated best %s to %f at batch %d",
                        cfg.best_metric, self.best_metric, batch_idx,
                    )
                    _metric_prefix = best_metric_keys[0].split("/")[0]  # "sft_eval_max"
                    metrics.update({
                        f"{_metric_prefix}/best_batch": batch_idx,
                        f"{_metric_prefix}/best_metric": self.best_metric,
                        f"{_metric_prefix}/best_sampling_client_path": best_sampling_client_path,
                    })
                    try:  # Saving replay buffer
                        self.replay_buffer.save_to_hf_dataset(self.best_replay_buffer_path)
                        logger.info("Saved best replay buffer to %s", self.best_replay_buffer_path)
                    except SchemaInferenceError:
                        logger.warning(
                            "Failed to save best replay buffer to %s\nIs replay buffer empty?",
                            self.best_replay_buffer_path
                        )
                sft_eval_loop_id += 1

            # 1. Sample rollouts for training
            self.env.split = "train"
            train_rollout_metrics, new_trajectories = await run_rollouts(
                sampling_client=sampling_client,
                renderer=renderer,
                hf_tokenizer=hf_tokenizer,
                generator_constructor=self.action_prompt_generator_constructor,
                env=self.env,
                cfg=cfg,
                batch_id=batch_idx,
                checkpoint_name=checkpoint_name,
                split="train",
                num_tries=cfg.num_tries,
                start_idx=batch_idx * cfg.batch_size,
                tasks_per_update=cfg.batch_size,
            )
            metrics.update(train_rollout_metrics)

            # Save replay buffer samples
            for trajectory in new_trajectories["policy"]:
                self.replay_buffer.add_trajectory(trajectory)
            self.replay_buffer.save_to_hf_dataset(self.last_replay_buffer_path)

            # 2. Update policy LLM with generated rollouts
            data_D, prepare_minibatch_metrics = await self.prepare_minibatch(
                new_trajectories=new_trajectories["policy"],
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
            self.ml_logger.log_metrics(metrics, step=batch_idx)

        return best_sampling_client_path
