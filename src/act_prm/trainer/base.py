"""
Parent class for Tinker trainers
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Callable

from omegaconf import DictConfig

import tinker
from tinker.types import LossFnType
from tinker_cookbook.utils import ml_log
from transformers import PreTrainedTokenizerBase

from ..environments import Environment
from ..generator.tinker import TinkerGenerator
# from ..replay_buffer.types import Trajectory

from .tinker.update import compute_full_batch_metrics_and_get_sampling_client, train_step
from .tinker.utils import timed


logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Parent class for SFT trainers
    """
    def __init__(
        self, 
        cfg: DictConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
        env: Environment,       # used to get training samples
        eval_env: Environment,  # could be the same as env, but update env.split
        ml_logger: ml_log.Logger,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs: Any,
    ) -> None:
        self.cfg = cfg
        self.training_client = training_client
        self.service_client = service_client
        self.env = env
        self.eval_env = eval_env
        self.ml_logger = ml_logger
        self.hf_tokenizer = hf_tokenizer

        self.best_metric = 1e8 if "loss" in cfg.best_metric else -1e8
        # self.best_sampling_client_path = ""

    @abstractmethod
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
        Implement synchronous training with Tinker
        """
        raise NotImplementedError

    @abstractmethod
    async def prepare_minibatch(self, **kwargs: Any) -> tuple[list[tinker.Datum], dict[str, Any]]:
        """
        Prepare a minibatch of trajectories for training
        """
        raise NotImplementedError

    # Modified from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L941
    async def do_train_step_and_get_sampling_client(
        self,
        batch_idx: int,
        training_client: tinker.TrainingClient,
        # service_client: tinker.ServiceClient,
        # new_trajectories: list[Trajectory],
        data_D: list[tinker.Datum],
        prepare_minibatch_metrics: dict[str, Any] | None = None,
        loss_fn: LossFnType | None = None,
        checkpoint_name: str | None = None,
        mini_batch_size: int | None = None,
        num_substeps: int | None = None,
    ) -> tuple[tinker.SamplingClient, dict[str, Any]]:
        """
        Update LLM policy with new trajectories and return updated sampling client
        """
        cfg = self.cfg
        loss_fn = loss_fn or cfg.loss_fn

        mini_batch_size = mini_batch_size or cfg.mini_batch_size
        num_substeps = num_substeps or cfg.num_substeps
        
        metrics = {}
        # Move the minibatch creation to outside this function
        prepare_minibatch_metrics = prepare_minibatch_metrics or {}
        # data_D, prepare_minibatch_metrics = await self.prepare_minibatch(
        #     new_trajectories=new_trajectories,
        #     service_client=service_client,
        #     model_name=cfg.model_name,
        #     kl_penalty_coef=cfg.kl_penalty_coef,
        #     kl_discount_factor=cfg.kl_discount_factor,
        # )
        metrics.update(prepare_minibatch_metrics)

        # Resample from data_D to determine actual training set
        if mini_batch_size and num_substeps is None:
            num_substeps = len(data_D) // mini_batch_size
        elif num_substeps:
            if mini_batch_size:
                # Potentially supersample data_D to hit mini_batch_size * num_substeps
                data_D = data_D[:mini_batch_size * num_substeps]
                while len(data_D) < mini_batch_size * num_substeps:
                    data_D.extend(random.sample(data_D, k=len(data_D)))
                data_D = data_D[:mini_batch_size * num_substeps]
        else:
            raise ValueError("Either mini_batch_size or num_substeps must be specified")
        # Randomly subsample training data if not evenly divisible by num_substeps (# of mini-batches)
        # -> Tinker requires this: https://tinker-docs.thinkingmachines.ai/rl/rl-hyperparams#multiple-updates-per-sampling-iteration
        if len(data_D) % num_substeps != 0:
            new_batch_size = (len(data_D) // num_substeps) * num_substeps
            data_D = random.sample(data_D, new_batch_size)
        random.shuffle(data_D)  # Ensure random ordering of mini-batches

        with timed("train", metrics):
            training_logprobs_D = await train_step(
                data_D,
                training_client,
                cfg.learning_rate,
                num_substeps,
                loss_fn,
            )
        sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
            training_client,
            batch_idx + 1,  # NOTE: saving the checkpoint as the i + 1 step
            data_D,
            training_logprobs_D,
            cfg.log_path,
            cfg.save_every,
            cfg.compute_post_kl,
            checkpoint_name=checkpoint_name,
        )
        metrics.update(full_batch_metrics)

        return sampling_client, metrics
