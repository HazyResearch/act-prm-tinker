"""
Tinker Trainers for fully synchronous Act-PRM training

Currently bloated logic that handles Act-PRM RL, SFT, and RL training

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


def display_metrics(metrics: dict[str, Any], title: str | None = None) -> None:
    # Display metrics
    table = Table(title=title, style="bright_yellow")
    # title=f"SFT Eval Loop {loop_id}, Eval {eval_idx}, Batch {batch_id}",
    # style="bright_yellow",
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="left")
    for k, v in metrics.items():
        table.add_row(k, str(v))
    console.print(table)


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
        # RL / Evaluation generator: does standard rollouts, see act_prm/generator/default.py
        self.rl_generator_constructor = self.get_generator_constructor(
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

        self.advantage_threshold = cfg.get("advantage_threshold", None)

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
        best_sft_state_path = ""
        best_rl_sampling_client_path = ""
        best_rl_state_path = ""

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
        if cfg.action_prompts:
            if cfg.num_batches_action_prompts > 0:
                logger.info("** Stage 1: Training RL action-prompted rollouts **")
                best_action_prompt_sampling_client_path = await self.do_rl_loop(
                    start_batch=0,
                    end_batch=cfg.num_batches_action_prompts,
                    cfg=cfg,
                    env=env,
                    eval_env=env,  # Evaluate thought-generation
                    generator_constructor=self.action_prompt_generator_constructor,
                    checkpoint_name="action_prompts",
                    name_or_identifier="Stage 1: RL Action-Prompted Rollouts",
                    # Save rollouts to a HF dataset
                    sampling_client=sampling_client,
                    renderer=renderer,
                    save_generator_constructor=self.action_prompt_generator_constructor,
                    save_env=env,  # We save rollouts from training env
                    save_name_or_identifier="Stage 1: SFT Generation with Act-PRM LLM",
                    trajectory_key="thought_action_policy",
                    dataset_prefix="mzio/aprm-sft_genthinkact",
                    dataset_suffix="-ap1",
                )
                # Get best action-prompted sampling client
                # sampling_client = self.service_client.create_sampling_client(
                sampling_client = await self.training_client.create_sampling_client_async(
                    model_path=best_action_prompt_sampling_client_path,
                )
            # Generate thought-action rollouts for all tasks in the ActPrmEnv
            # Will be size (len(env), group_size or sum([group_size * task_steps for task in env]))
            all_trajectories_per_env: list[list[Trajectory]] = await (
                # Inherited from trainer.rl.RLTrainer
                self.generate_and_save_trajectories(
                    sampling_client=sampling_client,
                    renderer=renderer,
                    save_generator_constructor=self.action_prompt_generator_constructor,
                    save_batch_idx=cfg.num_batches_action_prompts,
                    save_env=env,
                    save_name_or_identifier="Stage 1: SFT Generation with Act-PRM LLM",
                    trajectory_key="thought_action_policy",
                    dataset_prefix="mzio/aprm_sft_thought_action_rollouts",
                    dataset_suffix="-ap1_best",
                )
            )
            sft_group_size = cfg.group_size

            sft_batch_idx += cfg.num_batches_action_prompts  # so WandB logs correctly

        else:
            logger.info("** Stage 1: No action-prompted rollouts -> Using saved trajectories for SFT **")
            env.split = "train"
            env.default_context = []  # no few-shot
            # "Sample rollouts" from env samples
            # num_sft_gen_batches = min(len(env) // cfg.batch_size, 1)
            states: list[EnvironmentState] = await gather_with_progress(
                [
                    env.reset_async(sample_idx=idx, batch_idx=0)
                    for idx in range(len(env))
                ],
                desc="Initializing states for SFT training data from observations",
                colour="blue",
            )
            # all_trajectories_per_env: list[list[Trajectory]] = [
            #     self.get_trajectory_from_messages(
            #         messages=state.action_trajectory,
            #         hf_tokenizer=hf_tokenizer,
            #         system_prompt={"role": "system", "content": state.system_prompt},
            #         tools=state.tools,
            #     )
            #     for state in states
            # ]
            # hack
            # eval_env.split = "train"
            all_trajectories_per_env: list[list[Trajectory]] = await gather_with_progress(
                [
                    self.get_trajectory_from_actions_and_env(
                        state=state,
                        env=env,
                        hf_tokenizer=hf_tokenizer,
                    )
                    for state in states
                ],
                desc="Collecting SFT training data from actions and environment",
                colour="blue",
            )
            # To match amount of data to take num_substeps over, account for only having
            # 1 rollout per task
            sft_group_size = cfg.group_size * cfg.batch_size
            # long_trajs = [traj for group in all_trajectories_per_env for traj in group if len(traj.episode_steps) > 2]
            # rich_print(hf_tokenizer.decode(long_trajs[0].episode_steps[-1].state_action_tokens))
            # breakpoint()
            # all_trajectories_per_group = [group[1][trajectory_key] for group in all_trajectories_per_group]
            # Save thought-action rollouts so far to a HF Dataset and push to hub
            # -> Then can evaluate by seeing how training another LLM from scratch performs
            # -> Will overwrite existing dataset if it already exists (e.g., to hit total samples)
            
            dataset_prefix: str = "mzio/aprm-sft_act_only"
            ds_identifier = "-".join([
                f"{delim[:2].upper()}{self.run_name.split(delim)[-1].split("-")[0]}"
                for delim in ["enco=", "geco=", "se=", "re="]  # env, generator, seed, replicate
            ])
            ds_name = f"{dataset_prefix}-{ds_identifier}"
            if len(ds_name) > 96:   # hacky patch
                # HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. 
                # The name cannot start or end with '-' or '.' and the maximum length is 96
                ds_name = ds_name.replace("SE", "").replace("RE", "")
                if len(ds_name) > 96:
                    breakpoint()
            try:
                _trajectories = [traj[0] for traj in all_trajectories_per_env]
                _save_trajectories_to_hf_dataset(_trajectories, ds_name)
                _url = f"https://huggingface.co/datasets/{ds_name}"
                logger.info("Saved trajectories to HF Dataset: %s", _url)
            except Exception as e:
                _error_text = f"({type(e).__name__}: {e})"
                logger.error("Failed to save trajectories to HF Dataset: %s", _error_text)

        # ------------------------------------------------------------------------------
        # Second stage of training (SFT new policy LLM with the thought-action rollouts)
        # ------------------------------------------------------------------------------
        self.training_client = await self.service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )
        logger.info("** Stage 2: Training new policy LLM with %d SFT batches **", cfg.sft_num_batches)
        sft_best_metric = -float("inf")

        # Shuffle training data
        random.shuffle(all_trajectories_per_env)
        # Split into batches
        num_splits = max(len(all_trajectories_per_env) // sft_group_size, 1)
        all_train_batches: list[list[list[Trajectory]]] = split_list(
            all_trajectories_per_env, num_splits
        )
        sft_batch_start, sft_batch_end = sft_batch_idx, sft_batch_idx + cfg.sft_num_batches
        sft_pbar = tqdm(
            range(sft_batch_start, sft_batch_end),
            desc="Training new policy LLM with SFT (act_prm_sft_rl.train)",
            colour="blue",
            position=1,
        )
        for sft_batch_idx, overall_batch_idx in enumerate(sft_pbar):
            sft_metrics = {
                "progress/batch": sft_batch_idx,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (sft_batch_idx + 1) / cfg.sft_num_batches,
            }
            t_start = time.time()

            # Get training data
            if sft_batch_idx + 1 == num_splits:
                # Upon hitting our last available batch, shuffle and get new batches
                random.shuffle(all_trajectories_per_env)
                all_train_batches = split_list(all_trajectories_per_env, num_splits)
            
            adjusted_idx = sft_batch_idx % num_splits
            # (batch_size, group_size or total_episode_steps)
            trajectory_batches: list[list[Trajectory]] = all_train_batches[adjusted_idx]
            new_trajectories: list[Trajectory] = [
                trajectory for batch in trajectory_batches for trajectory in batch
            ]
            
            # Run evaluations
            if (
                cfg.sft_eval_every > 0 
                and (
                    (sft_batch_idx + 1) % cfg.sft_eval_every == 0 or
                    sft_batch_idx + 1 == cfg.sft_num_batches
                )
            ):
                with timed("run_evals", sft_metrics):
                    _num_eval_tasks = len(eval_env.datasets["eval"])
                    # Run on both train and eval splits
                    for _split in ["train", "eval"]:
                        setattr(eval_env, "split", _split)
                    
                        eval_env.split = _split
                        _name_or_identifier = (
                            "Stage 2: SFT Evaluation, "
                            f"Train Step {sft_batch_idx} / {cfg.sft_num_batches - 1}"
                            f" on {_split.capitalize()} split"
                        )
                        eval_rollout_metrics, _ = await run_rollouts(
                            sampling_client=sampling_client,
                            renderer=renderer,
                            hf_tokenizer=hf_tokenizer,
                            generator_constructor=self.rl_generator_constructor,
                            env=eval_env,
                            cfg=cfg,
                            batch_id=sft_batch_idx,
                            checkpoint_name="act_prm_sft",
                            split=_split,
                            num_tries=cfg.eval_num_tries,
                            start_idx=0,
                            tasks_per_update=_num_eval_tasks,
                            name_or_identifier=_name_or_identifier,
                        )
                        sft_metrics.update(eval_rollout_metrics)
                        display_metrics(
                            eval_rollout_metrics,
                            title=f"SFT Eval Loop {sft_batch_idx}, {_split.capitalize()} split"
                        )
                        if _split == "eval":
                            # Save best checkpoints -> Make sure this is on the eval split
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
                                best_sft_state_path = path_dict["state_path"]
                                logger.info("Saved best sampling client to %s", best_sft_sampling_client_path)
                                logger.info("Saved best state to %s", best_sft_state_path)
                                logger.info(
                                    "Updated best %s to %f at batch %d",
                                    cfg.best_metric, sft_best_metric, sft_batch_idx,
                                )
                                sft_metrics.update({
                                    f"{_metric_prefix}/best_batch": sft_batch_idx,
                                    f"{_metric_prefix}/best_metric": sft_best_metric,
                                    f"{_metric_prefix}/best_sampling_client_path": best_sft_sampling_client_path,
                                    f"{_metric_prefix}/best_state_path": best_sft_state_path,
                                })

            data_D, prepare_minibatch_metrics = self.prepare_sft_minibatch(
                new_trajectories=new_trajectories,
            )

            if sft_batch_idx % 100 == 0:
                _input_tokens = data_D[0].model_input.to_ints()
                rich_print(hf_tokenizer.decode(_input_tokens))

            sampling_client, update_metrics = await self.do_train_step_and_get_sampling_client(
                batch_idx=sft_batch_idx,
                training_client=self.training_client,
                data_D=data_D,
                prepare_minibatch_metrics=prepare_minibatch_metrics,
                loss_fn="cross_entropy",
                checkpoint_name="act_prm_sft",
                num_substeps=cfg.sft_num_substeps,  # 1
            )
            logger.info("Updated sampling client from SFT training, batch %d", sft_batch_idx)
            # Log metrics
            sft_metrics.update(update_metrics)
            sft_metrics["time/total"] = time.time() - t_start
            self.ml_logger.log_metrics(sft_metrics, step=overall_batch_idx)
            sft_pbar.set_postfix(**{k.split("/")[-1]: v for k, v in sft_metrics.items()})
        
        # ---------- Third stage of training (generate thoughts from states only) ----------
        logger.info("Stage 3 training RL")
        best_rl_metric = -float("inf")

        start_batch = overall_batch_idx
        end_batch = overall_batch_idx + end_batch
        num_batches = end_batch - start_batch

        if best_sft_state_path != "":
            logger.info("Attempting to load weights for RL from %s", best_sft_state_path)
            self.training_client = await (
                self.service_client.create_training_client_from_state_async(best_sft_state_path)
            )
            # MZ 1/18/26: 3 Questions:
            # 1. What's the advantage of `create_training_client_from_state_with_optimizer_async` instead?
            # 2. What's the diff b/t the service_client method(s) and `training_client.load_state_*`?
            # 3. This link suggests `training_client.load_state()` loads weights + optimizer:
            #    https://tinker-docs.thinkingmachines.ai/save-load#example-saving-to-resume-training
            #    But this link suggests it does not: 
            #    https://tinker-docs.thinkingmachines.ai/save-load#example-saving-to-resume-training 
            #    -> ¯\_(ツ)_/¯

            logger.info("Loaded RL weights from %s", best_sft_state_path)
            # Initialize new sampling client from best SFT state
            sampling_client, _ = await save_checkpoint_and_get_sampling_client(
                training_client=self.training_client,
                i_batch=start_batch,
                log_path=cfg.log_path,
                save_every=cfg.save_every,
                start_batch=start_batch,
            )
        
        rl_pbar = tqdm(
            range(start_batch, end_batch),
            desc="Training RL (act_prm_sft_rl.train)",
            colour="magenta",
            position=1,
        )
        for batch_idx, overall_batch_idx in enumerate(rl_pbar):
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
                    _name_or_identifier = (
                        f"Stage 3: RL Evaluation, Train Step {batch_idx} / {num_batches - 1}"
                    )
                    eval_rollout_metrics, _ = await run_rollouts(
                        sampling_client=sampling_client,
                        renderer=renderer,
                        hf_tokenizer=hf_tokenizer,
                        generator_constructor=self.rl_generator_constructor,
                        env=eval_env,
                        cfg=cfg,
                        batch_id=batch_idx,
                        checkpoint_name=checkpoint_name,
                        split="eval",
                        num_tries=cfg.eval_num_tries,
                        # Just use all eval tasks
                        start_idx=0,  
                        tasks_per_update=len(eval_env),
                        name_or_identifier=_name_or_identifier,
                    )
                    metrics.update(eval_rollout_metrics)
                    display_metrics(eval_rollout_metrics, title=f"RL Eval Loop {batch_idx}")

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
                    best_rl_state_path = path_dict["state_path"]
                    logger.info("Saved best sampling client to %s", best_rl_sampling_client_path)
                    logger.info(
                        "Updated best %s to %f at batch %d",
                        cfg.best_metric, best_rl_metric, batch_idx,
                    )
                    metrics.update({
                        f"{_metric_prefix}/best_batch": batch_idx,
                        f"{_metric_prefix}/best_metric": best_rl_metric,
                        f"{_metric_prefix}/best_sampling_client_path": best_rl_sampling_client_path,
                        f"{_metric_prefix}/best_state_path": best_rl_state_path,
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
            eval_env.split = "train"
            _name_or_identifier = (
                f"Stage 3: RL Training, Train Step {batch_idx} / {num_batches - 1}"
            )
            train_rollout_metrics, new_trajectories = await run_rollouts(
                sampling_client=sampling_client,
                renderer=renderer,
                hf_tokenizer=hf_tokenizer,
                generator_constructor=self.rl_generator_constructor,
                env=eval_env,
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
            self.ml_logger.log_metrics(metrics, step=overall_batch_idx)
            rl_pbar.set_postfix(**{k.split("/")[-1]: v for k, v in metrics.items()})

            logger.debug("TESTING RL batch %d, overall batch %d", batch_idx, overall_batch_idx)

        return best_rl_sampling_client_path


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
        **generate_and_save_trajectories_kwargs: Any,
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
            **generate_and_save_trajectories_kwargs
        )

    def prepare_sft_minibatch(
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
                    if self.advantage_threshold is not None:
                        # Only supervise if advantage > threshold
                        weight = float(weight > self.advantage_threshold)
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

    def get_trajectory_from_messages(
        self,
        messages: list[dict[str, str]],
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        system_prompt: dict[str, str] | None = None,
        tools: list[dict[str, str]] | None = None,
    ) -> list[Trajectory]:
        """
        Return singleton of a Trajectory where trajectory.episode_steps is a list 
        of StateActionSample (See replay_buffer.types)

        -> Assumes messages is a full chat dialogue of the form:
           [system_prompt, user_message, assistant_message, user_message, assistant_message, ...]
        -> But we hide prior observations (user_message) if self.hide_observations is True
        """
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        _tokenize_kwargs = {"enable_thinking": False, "tokenize": True, "tools": tools}

        episode_steps: list[StateActionSample] = []
        
        # Add system prompt to messages
        if messages[0]["role"] != "system":
            assert system_prompt is not None, (
                "System prompt must be provided if first message is not a system prompt"
            )
            messages = [system_prompt] + messages
        
        last_message_idx = 3
        # ^messages[:3] includes system_prompt, 1st user message, 1st assistant message
        while last_message_idx < len(messages):
            step_messages = self.maybe_hide_observations(messages[:last_message_idx])
            state_action_input_ids = hf_tokenizer.apply_chat_template(
                step_messages,
                add_generation_prompt=False,
                **_tokenize_kwargs,
            )
            state_len = len(hf_tokenizer.apply_chat_template(
                step_messages[:-1],
                add_generation_prompt=True,
                **_tokenize_kwargs,
            ))
            episode_steps.append(
                StateActionSample(  # simplified version of EpisodeStep
                    state_action_tokens=state_action_input_ids,
                    state_len=state_len,
                )
            )
            last_message_idx += 2
        
        trajectory = Trajectory(
            episode_steps=episode_steps,
            try_step=0,           # below are dummy values for SFT
            discount_factor=1.0,
            final_reward=1.0,
        )
        return [trajectory]

    async def get_trajectory_from_actions_and_env(
        self,
        state: EnvironmentState,
        env: Environment,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> list[Trajectory]:
        """
        Return singleton of a Trajectory where trajectory.episode_steps is a list 
        of StateActionSample (See replay_buffer.types)
        """
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer

        system_prompt = state.system_prompt
        
        # messages = state.new_messages
        # messages = [{"role": "system", "content": system_prompt}] + messages
        action_trajectory = state.action_trajectory
        tools = state.tools

        episode_steps: list[StateActionSample] = []
        _tokenize_kwargs = {"enable_thinking": False, "tokenize": True, "tools": tools}

        for action_idx, action in enumerate(action_trajectory):
            messages = (
                (state.prior_messages or []) 
                + (state.model_response or [])
                + state.new_messages
            )
            # Additional preprocessing into {"role": <role>, "content": <content>} format
            # -> See `act_prm.environments` classes for environment responses
            messages = [
                {"role": msg["role"], "content": msg["output"]}
                if msg.get("type", "") == "function_call_output"
                else msg
                for msg in messages                                                                                                         
            ] # Remove system prompt (will add it back after default context)
            if messages[0].get("role", "") == "system":
                messages = messages[1:]
            # Return final messages list
            messages = [
                {"role": "system", "content": state.base_env_state.system_prompt},
                *messages,
            ]
            # if isinstance(action, str):
            #     model_message = [{"role": "assistant", "content": action}]
            # else:
            #     model_message = [action]
            model_message = [{"role": "assistant", "content": ""}]  # thoughts
            parsed_actions = get_actions(model_message)
            env_step_result = await env.step_async(
                parsed_actions=parsed_actions,
                # model_response=model_message,
                current_state=state,
                current_messages=messages,
            )
            state = env_step_result.state
            # Update messages

            # try:
            #     step_messages = self.maybe_hide_observations(messages + model_message)
            # except Exception as e:
            #     print(messages, type(messages))
            #     print(model_message, type(model_message))
            #     breakpoint()
            step_messages = messages
            state_action_input_ids = hf_tokenizer.apply_chat_template(
                step_messages,
                add_generation_prompt=False,
                **_tokenize_kwargs,
            )
            state_len = len(hf_tokenizer.apply_chat_template(
                step_messages[:-1],
                add_generation_prompt=True,
                **_tokenize_kwargs,
            ))
            episode_steps.append(
                StateActionSample(  # simplified version of EpisodeStep
                    state_action_tokens=state_action_input_ids,
                    state_len=state_len,
                    state=step_messages[:-1],
                    action=step_messages[-1],
                )
            )
            # # Proceed to next step
            # messages = step_messages[:-1]
            
        
        trajectory = Trajectory(
            episode_steps=episode_steps,
            try_step=0,           # below are dummy values for SFT
            discount_factor=1.0,
            final_reward=1.0,
        )
        return [trajectory]
    
