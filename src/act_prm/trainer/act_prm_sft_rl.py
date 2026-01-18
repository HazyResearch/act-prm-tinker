"""
Tinker Trainers for fully synchronous Act-PRM training

```bash
uv run python main.py \
--is_async \
--env_config act_prm/browsecomp_100_hide_obs \
--eval_env_config browsecomp_plus/search_hide_obs \
--generator_config aprm_qwen3_ap \
--trainer_config qwen3_4b_aprm50_sft50_rl100 \
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
from .tinker.utils import save_checkpoint_and_get_sampling_client, timed
from .train import is_better, run_rollouts

logger = logging.getLogger(__name__)


def _save_trajectories_to_hf_dataset(
    trajectories: list[Trajectory],
    dataset_name: str,
    exclude_keys: list[str] | None = None,
) -> None:
    """
    Save a list of trajectories to a HF Dataset
    """
    exclude_keys = exclude_keys or ["state_action_tokens", "old_logprobs"]
    ds_samples = [
        {k: v for k, v in vars(step).items() if k not in exclude_keys}
        for trajectory in trajectories
        for step in trajectory.episode_steps
    ]  # Flatten to get dicts from list of EpisodeSteps in each Trajectory
    Dataset.from_list(ds_samples).push_to_hub(dataset_name, private=False)


def maybe_hide_observations(
    messages: list[dict[str, str]],
    hide_observations: bool = False,
    hidden_obs_content: str = "...",
    first_obs_to_show: int = 2,  # e.g., to keep prompt
    last_obs_to_show: int = 1,   # e.g., to keep last observation
) -> list[dict[str, str]]:
    """
    Hide past observations from messages
    """
    if not hide_observations:
        return messages

    user_indices = [
        idx for idx, message in enumerate(messages) if message["role"] in ["user", "tool"]
    ]
    last_message_idx = user_indices[-last_obs_to_show] if last_obs_to_show > 0 else len(messages)
    return [
        {"role": message["role"], "content": hidden_obs_content}
        if (
            message["role"] in ["user", "tool"]
            and (idx >= first_obs_to_show and idx < last_message_idx)
        )
        else message
        for idx, message in enumerate(messages)
    ]


class ActPrmSftRlTrainer(RLTrainer):
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
        # self.action_prompt_generator_constructor = self.get_generator_constructor(
        #     name="action_prompt_act_prm",
        #     keep_top_k=cfg.get("action_prompt_keep_top_k", None),
        #     **{k: v for k, v in generator_cfg.items() if k != "name"}
        # )
        self.action_prompt_generator_constructor = self.get_generator_constructor(
            **generator_cfg,
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
        name_or_identifier: str | None = None,
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
            name_or_identifier=name_or_identifier,
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
        best_sft_sampling_client_path = ""
        best_rl_sampling_client_path = ""

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
        sft_batch_idx = 0
        best_sft_sampling_client_path = ""
        
        if cfg.action_prompts:
            if cfg.num_batches_action_prompts > 0:
                logger.info("Stage 1: Training RL action-prompted rollouts")
                best_action_prompt_sampling_client_path = await self.do_rl_loop(
                    start_batch=0,
                    end_batch=cfg.num_batches_action_prompts,
                    cfg=cfg,
                    env=self.env,
                    eval_env=self.env,  # Evaluate thought-generation
                    generator_constructor=self.action_prompt_generator_constructor,
                    checkpoint_name="action_prompts",
                    name_or_identifier="Stage 1: RL Action-Prompted Rollouts",
                )
                # Get best action-prompted sampling client
                # sampling_client = self.service_client.create_sampling_client(
                sampling_client = await self.training_client.create_sampling_client_async(
                    model_path=best_action_prompt_sampling_client_path,
                )
            # Generate thought-action rollouts for all tasks in the ActPrmEnv
            self.env.split = "train"
            num_sft_gen_batches = len(self.env) // cfg.batch_size
            logger.info(
                "Generating thought-action rollouts for all %d tasks in the Act-PRM env, "
                "(%d batches for %d tasks per batch)",
                len(self.env), num_sft_gen_batches, cfg.batch_size,
            )
            all_trajectories_per_group: list[list[Trajectory]] = []
            for sft_gen_batch_idx in range(0, num_sft_gen_batches):
                # 1. Sample rollouts for training
                self.env.split = "train"
                _name_or_identifier = (
                    "Stage 2: SFT Generation with Act-PRM LLM, "
                    f"Batch {sft_gen_batch_idx} / {num_sft_gen_batches - 1}"
                )
                _, new_trajectories = await run_rollouts(
                    sampling_client=sampling_client,
                    renderer=renderer,
                    hf_tokenizer=hf_tokenizer,
                    generator_constructor=self.action_prompt_generator_constructor,
                    env=self.env,
                    cfg=cfg,
                    batch_id=sft_gen_batch_idx,
                    checkpoint_name="sft_gen",
                    split="train",
                    num_tries=cfg.num_tries,
                    start_idx=sft_gen_batch_idx * cfg.batch_size,
                    tasks_per_update=cfg.batch_size,
                    name_or_identifier=_name_or_identifier,
                )
                # all_new_trajectories.extend(new_trajectories["thought_action_policy"])
                all_trajectories_per_group.append(new_trajectories["thought_action_policy"])

                # Save thought-action rollouts so far to a HF Dataset and push to hub
                # -> Then can evaluate by seeing how training another LLM from scratch performs
                # -> Will overwrite existing dataset if it already exists (e.g., to hit total samples)
                _ds_name = []
                for delim in ["-enco=", "-geco=", "-se=", "-re="]:
                    _ds_name.append(self.run_name.split(delim)[-1].split("-")[0])
                _ds_name = "_".join(_ds_name)
                _ds_prefix = "mzio/aprm_sft_thought_action_rollouts"
                dataset_name = f"{_ds_prefix}-{_ds_name}-ap_rl{cfg.num_batches_action_prompts:04d}"
                try:
                    _traj_so_far = [traj for group in all_trajectories_per_group for traj in group]
                    _save_trajectories_to_hf_dataset(_traj_so_far, dataset_name)
                    logger.info("Saved thought-action rollouts to HF Dataset: %s", dataset_name)
                except Exception as e:
                    logger.error("Failed to save thought-action rollouts to HF Dataset: %s", e)

            # ------------------------------------------------------------------------------
            # Second stage of training (SFT new policy LLM with the thought-action rollouts)
            # ------------------------------------------------------------------------------
            self.training_client = await self.service_client.create_lora_training_client_async(
                cfg.model_name, rank=cfg.lora_rank
            )
            logger.info("Stage 2: Training new policy LLM with %d SFT batches", cfg.sft_num_batches)
            sft_best_metric = -float("inf")
            for sft_batch_idx in range(cfg.sft_num_batches):
                metrics = {
                    "progress/batch": sft_batch_idx,
                    "optim/lr": cfg.learning_rate,
                    "progress/done_frac": (sft_batch_idx + 1) / cfg.sft_num_batches,
                }
                t_start = time.time()
                
                # Run evaluations
                if cfg.sft_eval_every > 0 and sft_batch_idx % cfg.sft_eval_every == 0:
                    with timed("run_evals", metrics):
                        self.eval_env.split = "eval"
                        _name_or_identifier = (
                            "Stage 2: SFT Evaluation, "
                            f"Batch {sft_batch_idx} / {cfg.sft_num_batches - 1}"
                        )
                        eval_rollout_metrics, _ = await run_rollouts(
                            sampling_client=sampling_client,
                            renderer=renderer,
                            hf_tokenizer=hf_tokenizer,
                            generator_constructor=self.eval_generator_constructor,
                            env=self.eval_env,
                            cfg=cfg,
                            batch_id=sft_batch_idx,
                            checkpoint_name="act_prm_sft",
                            split="eval",
                            num_tries=cfg.eval_num_tries,
                            start_idx=0,
                            tasks_per_update=len(self.eval_env),
                            name_or_identifier=_name_or_identifier,
                        )
                        metrics.update(eval_rollout_metrics)
                    
                    # Save best checkpoints
                    _metric_prefix = "act_prm_sft_eval"
                    best_metric_key = f"{_metric_prefix}/try_{cfg.eval_num_tries-1}/{cfg.best_metric}"
                    last_metric = eval_rollout_metrics[best_metric_key]
                    best_ckpt_name = f"{_metric_prefix}_{sft_batch_idx:06d}_best"
                    if is_better(last_metric, sft_best_metric, cfg.best_metric):
                        sft_best_metric = last_metric
                        path_dict = await save_checkpoint_async(
                            training_client=self.training_client,
                            name=best_ckpt_name,
                            log_path=cfg.log_path,
                            loop_state={"batch": sft_batch_idx},
                            kind="both",
                        )
                        best_sft_sampling_client_path = path_dict["sampler_path"]
                        logger.info("Saved best sampling client to %s", path_dict["sampler_path"])
                        logger.info(
                            "Updated best %s to %f at batch %d",
                            cfg.best_metric, sft_best_metric, sft_batch_idx,
                        )
                        metrics.update({
                            f"{_metric_prefix}/best_batch": sft_batch_idx,
                            f"{_metric_prefix}/best_metric": sft_best_metric,
                            f"{_metric_prefix}/best_sampling_client_path": path_dict["sampler_path"],
                        })
                
                # Training updates
                if sft_batch_idx + 1 == num_sft_gen_batches:
                    random.shuffle(all_trajectories_per_group)
                adjusted_idx = sft_batch_idx % num_sft_gen_batches
                new_trajectories = all_trajectories_per_group[adjusted_idx]

                data_D, prepare_minibatch_metrics = await self.prepare_sft_minibatch(
                    new_trajectories=new_trajectories,
                )
                sampling_client, update_metrics = await self.do_train_step_and_get_sampling_client(
                    batch_idx=sft_batch_idx,
                    training_client=self.training_client,
                    data_D=data_D,
                    prepare_minibatch_metrics=prepare_minibatch_metrics,
                    loss_fn="cross_entropy",
                    checkpoint_name="act_prm_sft",
                    num_substeps=cfg.sft_num_substeps,
                )
                logger.info("Updated sampling client from SFT training, batch %d", sft_batch_idx)
                # Log metrics
                metrics.update(update_metrics)
                metrics["time/total"] = time.time() - t_start
                self.ml_logger.log_metrics(metrics, step=sft_batch_idx)
        
        # ---------- Third stage of training (generate thoughts from states only) ----------
        logger.info("Stage 3 training RL")
        best_rl_metric = -float("inf")

        start_batch = sft_batch_idx
        end_batch = sft_batch_idx + end_batch
        num_batches = end_batch - start_batch

        if best_sft_sampling_client_path != "":
            self.training_client = await self.service_client.create_training_client_from_state_async(
                best_sft_sampling_client_path
            )
            logger.info("Loaded RL weights from %s", best_sft_sampling_client_path)
            # Initial new sampling client
            sampling_client, _ = await save_checkpoint_and_get_sampling_client(
                training_client=self.training_client,
                i_batch=start_batch,
                log_path=cfg.log_path,
                save_every=cfg.save_every,
                start_batch=start_batch,
            )
        
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
                    _name_or_identifier = (
                        f"Stage 3: RL Evaluation, Batch {batch_idx} / {num_batches - 1}"
                    )
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
                        name_or_identifier=_name_or_identifier,
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
                if is_better(last_metric, best_rl_metric, cfg.best_metric):
                    best_rl_metric = last_metric
                    path_dict = await save_checkpoint_async(
                        training_client=self.training_client,
                        name=best_ckpt_name,
                        log_path=cfg.log_path,
                        loop_state={"batch": batch_idx},
                        kind="both",
                    )
                    best_rl_sampling_client_path = path_dict["sampler_path"]
                    logger.info("Saved best sampling client to %s", best_rl_sampling_client_path)
                    logger.info(
                        "Updated best %s to %f at batch %d",
                        cfg.best_metric, best_rl_metric, batch_idx,
                    )
                    metrics.update({
                        f"{_metric_prefix}/best_batch": batch_idx,
                        f"{_metric_prefix}/best_metric": best_rl_metric,
                        f"{_metric_prefix}/best_sampling_client_path": best_rl_sampling_client_path,
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
            self.eval_env.split = "train"
            _name_or_identifier = (
                f"Stage 3: RL Training, Batch {batch_idx} / {num_batches - 1}"
            )
            train_rollout_metrics, new_trajectories = await run_rollouts(
                sampling_client=sampling_client,
                renderer=renderer,
                hf_tokenizer=hf_tokenizer,
                generator_constructor=self.generator_constructor,
                env=self.eval_env,
                cfg=cfg,
                batch_id=batch_idx,
                checkpoint_name=checkpoint_name,
                split="train",
                num_tries=cfg.num_tries,
                start_idx=batch_idx * cfg.batch_size,
                tasks_per_update=cfg.batch_size,
                name_or_identifier=_name_or_identifier,
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

        return best_rl_sampling_client_path
