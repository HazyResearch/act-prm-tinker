"""
Tinker RL Trainer for fully synchronous on-policy training
"""

from os.path import join
from typing import Any, Callable
import logging
import time

import torch
from omegaconf import DictConfig

import tinker
from tinker import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.checkpoint_utils import save_checkpoint_async
from tinker_cookbook.utils import ml_log

from datasets.arrow_writer import SchemaInferenceError
from transformers import PreTrainedTokenizerBase

from ..environments import Environment
from ..generator import get_generator_constructor
from ..generator.tinker import TinkerGenerator
from ..replay_buffer import ReplayBuffer
from ..replay_buffer.types import Trajectory

from .base import BaseTrainer
from .tinker.metrics import incorporate_kl_penalty
from .tinker.utils import save_checkpoint_and_get_sampling_client, timed
from .train import is_better, run_rollouts


logger = logging.getLogger(__name__)


class RLTrainer(BaseTrainer):
    """
    Trainer for fully synchronous on-policy training with Tinker
    """
    def __init__(
        self, 
        cfg: DictConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
        # generator_constructor: Callable[..., TinkerGenerator],
        generator_cfg: DictConfig,
        replay_buffer: ReplayBuffer,
        env: Environment,
        eval_env: Environment,  # could be the same as env, but update env.split
        ml_logger: ml_log.Logger,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        super().__init__(
            cfg,
            training_client,
            service_client,
            env,
            eval_env,
            ml_logger,
            hf_tokenizer,
        )
        # Get constructor for LLM policy, determines how we generate rollouts
        self.generator_constructor = self.get_generator_constructor(**generator_cfg)
        self.replay_buffer = replay_buffer

        self.best_replay_buffer_path = join(cfg.checkpoint_path, "replay_buffer_best")
        self.last_replay_buffer_path = join(cfg.checkpoint_path, "replay_buffer")

    def get_generator_constructor(self, **kwargs: Any) -> Callable[..., TinkerGenerator]:
        """
        Get a (partially initialized) TinkerGenerator constructor by name
        """
        return get_generator_constructor(**kwargs, ml_logger=self.ml_logger)

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

        Returns the path to the best sampling path
        """
        cfg = cfg or self.cfg
        env = env or self.env
        eval_env = eval_env or self.eval_env
        generator_constructor = generator_constructor or self.generator_constructor

        best_sampling_client_path = ""  # Path to the best sampling client

        # Initial sampling client
        sampling_client, _ = await save_checkpoint_and_get_sampling_client(
            training_client=self.training_client,
            i_batch=start_batch,
            log_path=cfg.log_path,
            save_every=cfg.save_every,
            start_batch=start_batch,
            checkpoint_name=checkpoint_name,
        )

        model_name = cfg.model_name or self.training_client.get_info().model_data.model_name
        hf_tokenizer = self.hf_tokenizer or self.training_client.get_tokenizer()
        # ^Same as tinker_cookbook.tokenizer_utils.get_tokenizer(cfg.model_name)?
        renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(model_name)
        renderer = renderers.get_renderer(renderer_name, hf_tokenizer)
        logger.info("Using renderer: %s", renderer_name)

        num_batches = end_batch - start_batch
        for batch_idx in range(start_batch, end_batch):
            metrics = {
                "progress/batch": batch_idx,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (batch_idx + 1) / num_batches,
            }
            t_start = time.time()

            # Run evaluations
            if cfg.eval_every > 0 and batch_idx % cfg.eval_every == 0:
                with timed("run_evals", metrics):
                    eval_env.split = "eval"
                    eval_rollout_metrics, _ = await run_rollouts(
                        sampling_client=sampling_client,
                        renderer=renderer,
                        hf_tokenizer=hf_tokenizer,
                        generator_constructor=generator_constructor,
                        env=eval_env,
                        cfg=cfg,
                        batch_id=batch_idx,
                        checkpoint_name=checkpoint_name,
                        split="eval",
                        num_tries=cfg.eval_num_tries,
                        # Just use all eval tasks
                        start_idx=0,  
                        tasks_per_update=len(eval_env),
                    )
                    metrics.update(eval_rollout_metrics)

                # Save best checkpoints
                _metric_prefix = "eval" if checkpoint_name is None else f"{checkpoint_name}_eval"
                best_metric_key = f"{_metric_prefix}/try_{cfg.eval_num_tries-1}/{cfg.best_metric}"
                last_metric = eval_rollout_metrics[best_metric_key]
                best_ckpt_name = "best" if checkpoint_name is None else f"{checkpoint_name}_best"
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

            # 1. Sample rollouts for training
            env.split = "train"
            train_rollout_metrics, new_trajectories = await run_rollouts(
                sampling_client=sampling_client,
                renderer=renderer,
                hf_tokenizer=hf_tokenizer,
                generator_constructor=generator_constructor,
                env=env,
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

    async def prepare_minibatch(self, **kwargs: Any) -> tuple[list[tinker.Datum], dict[str, Any]]:
        """
        Prepare a minibatch of trajectories for training; see `_prepare_minibatch` for details
        """
        return await self._prepare_minibatch(**kwargs)

    async def _prepare_minibatch(
        self,
        new_trajectories: list[Trajectory],
        service_client: tinker.ServiceClient,
        model_name: str,
        kl_penalty_coef: float | None = None,
        kl_discount_factor: float | None = None,
    ) -> tuple[list[tinker.Datum], dict[str, Any]]:
        """
        Subclass implementation of `prepare_minibatch`
        """
        metrics = {}
        # Assemble training data
        with timed("assemble_training_data", metrics):
            data_D: list[tinker.Datum] = []
            metadata_D: list[dict[str, int]] = []
            for trajectory in new_trajectories:
                for episode_step in trajectory.episode_steps:
                    sa_input_ids = episode_step.state_action_tokens
                    act_logprobs = episode_step.old_logprobs
                    input_tokens = sa_input_ids[:-1]
                    target_tokens = sa_input_ids[1:]
                    target_state_len = episode_step.state_len - 1

                    padded_logprobs = [0.0] * target_state_len + act_logprobs
                    adv = episode_step.advantage
                    padded_advantages = [0.0] * target_state_len + [adv] * len(act_logprobs)
                    padded_mask = [0.0] * target_state_len + [1.0] * len(act_logprobs)

                    try:
                        assert (
                            len(input_tokens)
                            == len(padded_logprobs)
                            == len(padded_advantages)
                            == len(target_tokens)
                        )
                    except AssertionError:
                        # print(self.hf_tokenizer.decode(target_tokens))
                        # print(self.hf_tokenizer.decode(target_tokens[target_state_len:]))
                        logger.error(f"len(input_tokens): {len(input_tokens)}")
                        logger.error(f"len(padded_logprobs): {len(padded_logprobs)}")
                        logger.error(f"len(padded_advantages): {len(padded_advantages)}")
                        logger.error(f"len(target_tokens): {len(target_tokens)}")
                        breakpoint()

                    metadata_D.append({
                        "sample_id": episode_step.unique_data_sample_id,
                        "generation_id": episode_step.generation_id,
                    })
                    data_D.append(
                        tinker.Datum(
                            model_input=tinker.ModelInput.from_ints(input_tokens),
                            loss_fn_inputs={
                                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                                "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                                "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                                "mask": TensorData.from_torch(torch.tensor(padded_mask)),  # for KL
                            },
                        )
                    )
        # Incorporate KL penalty if configured
        # - Copied from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L763
        if kl_penalty_coef > 0:
            with timed("kl_vs_base", metrics):
                kl_penalty_metrics = await incorporate_kl_penalty(
                    data_D,
                    service_client.create_sampling_client(base_model=model_name),
                    # ^^^ TODO: replace with the model we load, if relevant
                    kl_penalty_coef,
                    kl_discount_factor,
                )
            metrics.update(kl_penalty_metrics)

        return data_D, metrics
