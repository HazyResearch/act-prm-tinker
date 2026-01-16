"""
Tinker SFT Trainer, with sampling-based evaluation on an environment
"""

import asyncio
import logging
import random
import time
from typing import Any, Callable

import torch
from omegaconf import DictConfig

import tinker
from tinker import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.checkpoint_utils import save_checkpoint_async
from transformers import PreTrainedTokenizerBase

from ..environments import Environment, EnvironmentState
from ..generator.tinker import TinkerGenerator
from ..replay_buffer.types import StateActionSample, Trajectory

from .base import BaseTrainer
from .tinker.utils import save_checkpoint_and_get_sampling_client, timed
from .train import is_better, run_rollouts


logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    """
    Trainer for supervised fine-tuning (SFT) with Tinker
    """
    def _get_trajectory_from_messages(
        self,
        messages: list[dict[str, str]],
        hf_tokenizer: PreTrainedTokenizerBase,
        system_prompt: dict[str, str] | None = None,
        tools: list[dict[str, str]] | None = None,
    ) -> Trajectory:
        """
        Get a Trajectory where trajectory.episode_steps is a list of StateActionSample
        (See replay_buffer.types)
        """
        episode_steps: list[StateActionSample] = []
        _tokenize_kwargs = {"tokenize": True, "tools": tools}

        # Add system prompt to messages
        if messages[0]["role"] != "system":
            assert system_prompt is not None, (
                "System prompt must be provided if first message is not a system prompt"
            )
            messages = [system_prompt] + messages
        
        last_message_idx = 3  # messages[:3] includes system_prompt, user_message, first assistant_message
        while last_message_idx < len(messages):
            state_action_input_ids = hf_tokenizer.apply_chat_template(
                messages[:last_message_idx],
                add_generation_prompt=False,
                **_tokenize_kwargs,
            )
            state_len = len(hf_tokenizer.apply_chat_template(
                messages[:last_message_idx - 1],
                add_generation_prompt=True,
                **_tokenize_kwargs,
            ))
            episode_steps.append(
                StateActionSample(
                    state_action_tokens=state_action_input_ids,
                    state_len=state_len,
                )
            )
            last_message_idx += 2
        
        return Trajectory(
            episode_steps=episode_steps,
            try_step=0,           # below are dummy values for SFT
            discount_factor=1.0,
            final_reward=1.0,
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
    ) -> str:
        """
        Implement supervised fine-tuning (SFT) with Tinker

        For each training batch:
        1. Loads the most recent checkpoint and determines how we generate rollouts
        2. Generates rollouts (optionally running on the evaluation environment)
        3. Performs a policy update
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
        # Task indices from demonstrations we can sample
        indices_to_sample = list(range(len(env)))
        random.shuffle(indices_to_sample)

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
                    metrics.update({
                        f"{_metric_prefix}/best_batch": batch_idx,
                        f"{_metric_prefix}/best_metric": self.best_metric,
                        f"{_metric_prefix}/best_sampling_client_path": best_sampling_client_path,
                    })

            # 1. "Sample rollouts" for training
            env.split = "train"
            # Here we iterate through the task demonstrations in the environment
            # -> To match training samples vs RL, we set the total number of trajectories
            #    as cfg.batch_size * cfg.group_size (i.e., tasks_per_update * num_return_sequences)
            num_trajectories = cfg.batch_size * cfg.group_size
            if len(indices_to_sample) < num_trajectories:  # Reset and shuffle indices_to_sample
                indices_to_sample = list(range(len(env)))
                random.shuffle(indices_to_sample)

            sampled_indices = [indices_to_sample.pop() for _ in range(num_trajectories)]
            states: list[EnvironmentState] = await asyncio.gather(
                *[
                    env.reset_async(sample_idx=idx, batch_idx=batch_idx)
                    for idx in sampled_indices
                ]
            )
            new_trajectories = [
                self._get_trajectory_from_messages(
                    messages=state.action_trajectory,
                    hf_tokenizer=hf_tokenizer,
                    system_prompt={"role": "system", "content": state.system_prompt},
                    tools=state.tools,
                )
                for state in states
            ]
            data_D, prepare_minibatch_metrics = await self.prepare_minibatch(
                new_trajectories=new_trajectories,
            )

            # 2. Update policy LLM with generated rollouts
            sampling_client, update_metrics = await self.do_train_step_and_get_sampling_client(
                batch_idx=batch_idx,
                training_client=self.training_client,
                data_D=data_D,
                prepare_minibatch_metrics=prepare_minibatch_metrics,
                loss_fn="cross_entropy",
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
    ) -> tuple[list[tinker.Datum], dict[str, Any]]:
        """
        Subclass implementation of `prepare_minibatch`
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
