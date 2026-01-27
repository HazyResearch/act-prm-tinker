"""
Main script for training + evaluating LLMs using PyTorch

Example command:
```bash
uv run python main_pytorch_sft.py \
--is_async \
--env_config hotpotqa_mc/fewshot2 \
--generator_config default \
--trainer_config qwen3_4b_pg \
--replay_buffer_config default \
--log_path ./logs \
--model_name Qwen/Qwen3-4B-Instruct-2507 \
--lora_rank 32 \
--seed 42 --replicate 0 --verbose
```
"""

import argparse
import logging
import sys
from typing import Any

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from rich import print as rich_print

from tinker_cookbook.utils import ml_log  # still use nice logging

from act_prm.environments import get_env
from act_prm.llm_handlers import load_llm
from act_prm.llm_handlers.huggingface import HuggingFaceLLM
from act_prm.lora import display_trainable_parameter_count, get_lora_model
from act_prm.pytorch import get_optimizer, SftTrainer
from act_prm.replay_buffer import get_replay_buffer
from act_prm.utils import get_args, print_config, seed_everything
# from act_prm.trainer import get_trainer


logger = logging.getLogger(__name__)


def update_configs(
    args: argparse.Namespace,
    *configs: DictConfig | None,
) -> tuple[DictConfig | None, ...]:
    """
    Update configs with any specified + applicable command-line arguments
    """
    # A bit heinous, but loop through all configs to update any applicable args
    for config in configs:
        if config is not None:
            for argname, argval in vars(args).items():
                if argval is not None and argname in config and argname != "model_config":
                    config[argname] = argval
    return configs


def get_param_color(param_name: str) -> str:
    if "self_attn" in param_name:
        return "yellow"
    elif "mlp" in param_name:
        return "cyan"


def main() -> None:
    """
    Main training function
    """
    # Initialize experiment
    args = get_args()
    seed_everything(args.seed)
    load_dotenv()  # Setup environment variables from .env file

    # Get default configs
    model_cfg         = OmegaConf.load(f"./configs/model/{args.model_config}.yaml")
    lora_cfg          = OmegaConf.load(f"./configs/lora/{args.lora_config}.yaml")
    env_cfg           = OmegaConf.load(f"./configs/environments/{args.env_config}.yaml")
    # generator_cfg     = OmegaConf.load(f"./configs/generator/{args.generator_config}.yaml")
    generator_cfg     = None
    trainer_cfg       = OmegaConf.load(f"./configs/trainer/{args.trainer_config}.yaml")
    replay_buffer_cfg = OmegaConf.load(f"./configs/replay_buffer/{args.replay_buffer_config}.yaml")
    
    # Optional environment configs
    eval_env_cfg = env_cfg
    base_env_cfg = None
    if args.eval_env_config is not None:
        eval_env_cfg = OmegaConf.load(f"./configs/environments/{args.eval_env_config}.yaml")
    if args.base_env_config is not None:
        base_env_cfg = OmegaConf.load(f"./configs/environments/{args.base_env_config}.yaml")
    # Update configs from args
    updated_cfgs = update_configs(
        args, model_cfg, lora_cfg,
        env_cfg, eval_env_cfg, base_env_cfg,
        generator_cfg, trainer_cfg, replay_buffer_cfg,
    )
    if args.verbose:
        cfg_names = [
            "model", "lora", "env", "eval_env", "base_env", "generator", "trainer", "replay_buffer"
        ]
        for cfg, cfg_name in zip(updated_cfgs, cfg_names):
            if cfg is not None:
                print_config(cfg, cfg_name.upper())
    # Get updated config variables
    model_cfg, lora_cfg = updated_cfgs[:2]
    env_cfg, eval_env_cfg, base_env_cfg, generator_cfg, trainer_cfg, replay_buffer_cfg = updated_cfgs[2:]
    # Make env_cfg tokenizers consistent with llm.tokenizer
    env_cfg.pretrained_model_config = model_cfg.model_config
    eval_env_cfg.pretrained_model_config = model_cfg.model_config
    
    cfg = trainer_cfg  # Main config to reference (has all Tinker training attributes)
    cfg.run_name = args.run_name
    cfg.lora_checkpoint_path = args.lora_checkpoint_path

    # Setup logging to WandB
    cfg_for_logger: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    cfg_for_logger.update(vars(args))  # WandB parseable, add argparse args to log
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        config=cfg_for_logger,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)
    rich_print(
        "-> [bright_yellow]Saving LoRA checkpoints to [/bright_yellow]"
        f"[bright_blue]{cfg.lora_checkpoint_path}[/bright_blue]"
    )

    # Get LLM and attach LoRAs
    llm: HuggingFaceLLM = load_llm(**model_cfg)
    llm.model = get_lora_model(llm.model, **lora_cfg)
    # save_lora(llm.model, cfg.lora_checkpoint_path)
    optimizer = get_optimizer(llm.model, learning_rate=cfg.learning_rate)  # simple for now
    
    if args.verbose:  # Display trainable parameters
        _params_text = "Trainable parameters:\n"
        _param_names = [n for n, p in llm.model.named_parameters() if p.requires_grad]
        _params_text += "\n".join(
            f"├── [{get_param_color(n)}]{n}[/{get_param_color(n)}]" for n in _param_names
        )
        rich_print(_params_text)
    display_trainable_parameter_count(llm.model)
    
    replay_buffer = get_replay_buffer(**replay_buffer_cfg)

    # Get environment, replay buffer, and generator class
    base_env = get_env(**base_env_cfg) if base_env_cfg is not None else None
    env = get_env(**env_cfg, base_env=base_env)  # For ActPrmEnvWithBaseEnv
    # Reuse env if eval_env not specified; we always specify the split for loading new tasks
    eval_env = get_env(**eval_env_cfg) if args.eval_env_config else env

    env.tokenizer = llm.tokenizer
    eval_env.tokenizer = llm.tokenizer
    if base_env is not None:
        base_env.tokenizer = llm.tokenizer

    # Make identifiers identifiable
    cfg.run_url = ml_logger.get_logger_url() if ml_logger is not None else None
    cfg.run_cmd = " ".join(sys.argv)

    for attr in ["run_url", "run_cmd"]:
        for env in [env, eval_env, base_env]:
            if env is not None:
                setattr(env, attr, cfg.get(attr))

    # Training loop
    trainer = SftTrainer(
        cfg=cfg,
        llm=llm,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        env=env,
        eval_env=eval_env,
        ml_logger=ml_logger,
        hf_tokenizer=llm.tokenizer,
        checkpoint_path=cfg.lora_checkpoint_path,
    )
    trainer.train()

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
