"""
Tinker Trainers for fully synchronous Act-PRM training
"""

import logging
import time
from os.path import join
from typing import Any, Callable

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
from ..environments.act_prm import ActPrmEnv
from ..generator.tinker import TinkerGenerator
from ..replay_buffer import ReplayBuffer
from ..replay_buffer.types import Trajectory

from .rl import RLTrainer
from .tinker.utils import save_checkpoint_and_get_sampling_client, timed
from .train import is_better, run_rollouts

logger = logging.getLogger(__name__)


class ActPrmTrainer(RLTrainer):
    """
    Trainer for fully synchronous Act-PRM training
    """
    def __init__(
        self, 
        cfg: DictConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
        # generator_constructor: Callable[..., TinkerGenerator],
        generator_cfg: DictConfig,
        replay_buffer: ReplayBuffer,
        env: ActPrmEnv,
        eval_env: Environment,
        ml_logger: ml_log.Logger,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
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
        )
        # Evaluation generator does standard rollouts, see act_prm/generator/default.py
        self.eval_generator_constructor = self.get_generator_constructor(
            name="default",
            verbose=generator_cfg.verbose,
        )
        self.action_prompt_generator_constructor = self.get_generator_constructor(
            name="action_prompt_act_prm",
            **{k: v for k, v in generator_cfg.items() if k != "name"}
        )

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

    async def prepare_sft_minibatch(
        self,
        new_trajectories: list[Trajectory],
    ) -> tuple[list[tinker.Datum], dict[str, Any]]:
        """
        Prepare a minibatch of trajectories for SFT training
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
                    weights = [0.0] * target_state_len + [1.0] * target_action_len
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
        best_action_prompt_sampling_client_path = ""
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

        # ---------- First stage of training (include actions in prompts) ----------
        # if cfg.action_first_prompts is True, we first train with action-first prompts
        # via TinkerActionPromptActPrmGenerator (via self.action_prompt_generator_constructor)
        if cfg.action_prompts:
            if cfg.num_batches_action_prompts > 0:
                best_action_prompt_sampling_client_path = await self.do_rl_loop(
                    start_batch=0,
                    end_batch=cfg.num_batches_action_prompts,
                    cfg=cfg,
                    env=self.env,
                    eval_env=self.eval_env,  # Evaluate thought-generation
                    generator_constructor=self.action_prompt_generator_constructor,
                    checkpoint_name="action_prompts",
                )
                # Get best action-prompted sampling client
                # sampling_client = await self.service_client.create_sampling_client(
                sampling_client = await self.training_client.create_sampling_client(
                    model_path=best_action_prompt_sampling_client_path,
                )
            # Generate thought-action rollouts for all tasks in the ActPrmEnv
            _end_batch = len(self.env) // cfg.batch_size
            all_new_trajectories: list[Trajectory] = []
            for batch_idx in range(0, _end_batch):
                # 1. Sample rollouts for training
                self.env.split = "train"
                _, new_trajectories = await run_rollouts(
                    sampling_client=sampling_client,
                    renderer=renderer,
                    hf_tokenizer=hf_tokenizer,
                    generator_constructor=self.action_prompt_generator_constructor,
                    env=self.env,
                    cfg=cfg,
                    batch_id=batch_idx,
                    checkpoint_name="action_prompts",
                    split="train",
                    num_tries=cfg.num_tries,
                    start_idx=batch_idx * cfg.batch_size,
                    tasks_per_update=cfg.batch_size,
                )
                all_new_trajectories.extend(new_trajectories["thought_action_policy"])

            # Train new policy LLM with the thought-action rollouts
            self.training_client = await self.service_client.create_lora_training_client_async(
                cfg.model_name, rank=cfg.lora_rank
            )
            logger.info(
                "Training new policy LLM with %d thought-action rollouts",
                len(all_new_trajectories),
            )
            data_D, prepare_minibatch_metrics = await self.prepare_sft_minibatch(
                new_trajectories=all_new_trajectories,
            )
            # New sampling client from SFT training
            sampling_client, update_metrics = await self.do_train_step_and_get_sampling_client(
                batch_idx=0,
                training_client=self.training_client,
                data_D=data_D,
                prepare_minibatch_metrics=prepare_minibatch_metrics,
                loss_fn="cross_entropy",
                checkpoint_name="action_prompts",
                mini_batch_size=cfg.action_prompt_mini_batch_size,
                num_substeps=cfg.action_prompt_num_substeps,
            )
            logger.info("Updated sampling client from SFT training")
        
        # ---------- Second stage of training (generate thoughts from states only) ----------
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
                    self.eval_env.split = "eval"
                    eval_rollout_metrics, _ = await run_rollouts(
                        sampling_client=sampling_client,
                        renderer=renderer,
                        hf_tokenizer=hf_tokenizer,
                        generator_constructor=self.eval_generator_constructor,
                        env=self.eval_env,
                        cfg=cfg,
                        batch_id=batch_idx,
                        checkpoint_name=checkpoint_name,
                        split="eval",
                        num_tries=cfg.eval_num_tries,
                        # Just use all eval tasks
                        start_idx=0,  
                        tasks_per_update=len(self.eval_env),
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
            self.env.split = "train"
            train_rollout_metrics, new_trajectories = await run_rollouts(
                sampling_client=sampling_client,
                renderer=renderer,
                hf_tokenizer=hf_tokenizer,
                generator_constructor=self.generator_constructor,
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
