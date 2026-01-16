# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Act-PRM is a framework for training Action Process Reward Models using reinforcement learning with Tinker (a remote training service). The codebase implements policy gradient methods for training LLMs on agentic tasks involving tool use.

## Commands

### Setup
```bash
# Install uv package manager: https://docs.astral.sh/uv/installation/
# Install dependencies
uv sync
```

### Running Experiments
```bash
# Basic training command structure
uv run python main.py \
  --is_async \
  --env_config <env_name>/<config_name> \
  --generator_config <default|grpo> \
  --trainer_config qwen3_4b_pg \
  --replay_buffer_config default \
  --log_path ./logs \
  --model_name <model_path> \
  --lora_rank 32 \
  --seed 42 --replicate 0 --verbose

# Example: HotpotQA Multiple Choice
CUDA_VISIBLE_DEVICES=0 uv run python main.py \
  --is_async --env_config hotpotqa_mc/fewshot2 \
  --generator_config default --trainer_config qwen3_4b_pg \
  --replay_buffer_config default --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --lora_rank 32 --seed 42 --replicate 0 --verbose
```

### Environment Variables
Create a `.env` file with:
- `TINKER_API_KEY` - Required for Tinker training service
- `HF_TOKEN` - HuggingFace token
- `WANDB_API_KEY` - Weights & Biases logging
- `WANDB_ENTITY` - WandB entity name

## Architecture

### Core Training Loop (`main.py` → `trainer/train.py` / `trainer/rl.py`)
1. Loads configs from `configs/` (environment, generator, trainer, replay_buffer)
2. Creates a Tinker training client (LoRA, checkpoint resume, or weight-only load)
3. Runs fully synchronous training via `do_sync_training()` (or `RLTrainer.train()`), which
   alternates rollout generation and policy updates, with eval checkpoints on schedule

### Key Components

**Environments** (`src/act_prm/environments/`):
- Base class `Environment` defines `reset()`, `step()`, `shuffle()` interface
- Each environment has sync and async variants (e.g., `HotpotQAMultipleChoiceEnv`, `AsyncHotpotQAMultipleChoiceEnv`)
- Available: `hotpotqa_mc`, `browsecomp_plus_search`, `longbench_v2`
- Tools inherit from `BaseTool` with `__call__()` and `get_tool_desc()` methods

**Generator** (`src/act_prm/generator/`):
- `TinkerGenerator.do_single_rollout()`: runs full episode, collects EpisodeSteps
- `TinkerGenerator.do_group_rollout()`: generates multiple trajectories for a sample
- `TinkerGRPOGenerator`: uses mean-centered returns for GRPO-style training
- `TinkerActPrmGenerator`: step-wise thought sampling with per-step rewards for Act-PRM
- `TinkerActionPromptActPrmGenerator`: action-prompted thought rollouts (adds `"act_prompt"` groups)

**Replay Buffer Types** (`src/act_prm/replay_buffer/types.py`):
- `EpisodeStep`: single (state, action, next_obs) transition with tokens and logprobs
- `Trajectory`: sequence of EpisodeSteps from one rollout
- `TrajectoryGroup`: batch of Trajectories for computing advantages
- `MeanCenteredTrajectoryGroup`: GRPO-style advantage computation (final_reward - mean)
**Trainer** (`src/act_prm/trainer/`):
- `train.py`: core async helpers for rollouts, minibatch prep, and train steps
- `rl.py`: `RLTrainer` orchestrates synchronous on-policy training
- `tinker/`: Tinker-specific metrics, update logic, and checkpoint helpers


**LLM Handlers** (`src/act_prm/llm_handlers/`):
- `TinkerCompleter`: wraps Tinker SamplingClient for generation and logprob computation
- `action_utils.get_actions()`: parses tool calls from model output

### Config Structure
- `configs/environments/<env>/<variant>.yaml` - environment settings
- `configs/generator/{default,grpo}.yaml` - generator type selection
- `configs/trainer/qwen3_4b_pg.yaml` - training hyperparameters (batch_size, group_size, learning_rate, etc.)
- `configs/replay_buffer/default.yaml` - replay buffer settings

### Training Flow
1. `run_rollouts()` generates trajectories using `TinkerGenerator`/Act-PRM variants
2. Each trajectory contains `EpisodeStep`s with state/action tokens and logprobs
3. `prepare_minibatch()` assembles training data with advantages (and optional KL penalty)
4. `train_step()` performs policy gradient updates via Tinker
5. Replay buffers are saved under `checkpoint_path` (`replay_buffer` + `replay_buffer_best`)
6. Checkpoints saved to `{log_path}/{env_config}/{model_name}/{run_name}/` (plus best on eval)
