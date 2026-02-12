# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Python version**: 3.12 (see `.python-version`)

## Project Overview

Act-PRM (Action Process Reward Models) is an experimental codebase for training LLMs on agentic tasks using various training paradigms: Supervised Fine-Tuning (SFT), Reinforcement Learning (Policy Gradient), and Act-PRM training. The project supports two parallel implementations: Tinker-based (async, proprietary backend) and pure PyTorch (sync, self-contained).

## Commands

```bash
# Install dependencies
uv sync

# Install FlashAttention-2 (optional, for performance)
uv pip install flash-attn --no-build-isolation-package flash-attn

# Run with Tinker backend (async)
uv run python main.py \
  --is_async \
  --env_config <env_config> \
  --generator_config <generator_config> \
  --trainer_config <trainer_config> \
  --replay_buffer_config default \
  --model_name <model_name> \
  --lora_rank 32

# Run with PyTorch backend (sync)
uv run python main_pytorch.py \
  --env_config <env_config> \
  --model_config <model_config> \
  --lora_config <lora_config> \
  --generator_config <generator_config> \
  --trainer_config <trainer_config> \
  --replay_buffer_config default
```

No formal test suite exists. Development relies on experiment runs with WandB logging.

## Architecture

### Entry Points
- `main.py` - Tinker-based training (async, uses Tinker SamplingClient)
- `main_pytorch.py` - Pure PyTorch training (sync, HuggingFace + PEFT)
- `main_pytorch_lm.py` - Language model variant

### Source Code (`src/act_prm/`)

**Environments** (`environments/`): Task environments that agents interact with
- Base abstract class with async/sync variants
- Implementations: `act_prm`, `hotpotqa_mc`, `browsecomp_plus`, `textworld`, `longbench_v2`, `babyai_text`, `action_lm`

**Trainers**: Two parallel implementations
- Tinker-based (`trainer/`): `SFTTrainer`, `RLTrainer`, `ActPrmTrainer`, `ActPrmSftEvalTrainer`, `ActPrmSftRlTrainer`, `ActPrmJointTrainer`
- PyTorch-based (`pytorch/trainer/`): `SftTrainer`, `RLTrainer`, `ActPrmForSftTrainer`

**Generators** (`generator/`, `pytorch/generator/`): Rollout generation strategies
- `TinkerGenerator`, `TinkerActPrmGenerator`, `TinkerActionPromptActPrmGenerator`
- PyTorch variants for sync operation

**LLM Handlers** (`llm_handlers/`): Abstractions for different LLM backends (Tinker, HuggingFace, OpenAI)

**Replay Buffer** (`replay_buffer/`): Trajectory storage with grouping and mean-centering

### Configuration System

Configs are in `configs/` organized by component type:
- `environments/` - Task environment configs (includes `babyai/` for direct BabyAI and `act_prm/babyai.yaml` for Act-PRM-wrapped BabyAI)
- `generator/` - Rollout generation configs
- `trainer/` - Training algorithm configs (SFT, RL, Act-PRM variants)
- `model/` - Model architecture configs
- `lora/` - LoRA fine-tuning configs (rank, alpha, target modules)
- `replay_buffer/` - Replay buffer configs

YAML configs compose via OmegaConf. CLI arguments override YAML defaults.

### Factory Pattern

All major components use factory functions in their respective `__init__.py`:
- `get_env(name, **kwargs)`
- `get_trainer(name, **kwargs)`
- `get_replay_buffer(name, **kwargs)`
- `get_generator_constructor(name, **kwargs)`

New implementations require adding a factory entry.

## Key Training Flags

- `--actions_only` - Train on actions only (no thoughts)
- `--hide_observations` - Hide past observations from context
- `--mean_center` - Mean-center returns (GRPO-like)
- `--reward_method em|action_probs` - Reward computation method
- `--batch_size`, `--group_size`, `--num_substeps` - Control minibatch composition

## Branch Structure

The codebase is split across branches:
- **`main`** — Tinker-based (async) training infrastructure
- **`pytorch`** — Pure PyTorch training infrastructure (`main_pytorch.py`, `src/act_prm/pytorch/`, lora/model/trainer configs)
- **`madison/babyai`** — BabyAI environment files and configs (branched from `main`, does **not** contain PyTorch infra)

To run BabyAI + PyTorch training, you need code from both `pytorch` and `madison/babyai`. The Colab notebook handles this by checking out `pytorch` and cherry-picking BabyAI files. See the notebook setup cell for details.

## BabyAI-Text Environment

The `babyai_text` environment wraps [BabyAI-Text](https://github.com/flowersteam/Grounding_LLMs_with_online_RL) grid-world levels into the act-prm framework. It provides both sync (`BabyAiTextEnv`) and async (`AsyncBabyAiTextEnv`) variants, registered in the factory as `babyai_text`. The environment converts 6 discrete BabyAI actions (turn left/right, go forward, pick up, drop, toggle) into LLM tool-call format and uses the BabyAI `Bot` planner to generate gold demonstration trajectories at reset time.

Tool definitions are passed via `state.tools` (not embedded in the system prompt), so `apply_chat_template(tools=state.tools, ...)` formats them in each model's native chat template. This lets the same environment work across Llama, Qwen, DeepSeek, etc. without prompt changes.

**Configs:**
- `configs/environments/babyai/` — Direct BabyAI env configs (`default.yaml`, `eval.yaml`, `test.yaml`), using `BabyAI-MixedTestLocal-v0` with 20 max turns
- `configs/environments/act_prm/babyai.yaml` — Act-PRM wrapper over BabyAI (for thought-generation training on top of BabyAI actions)

**Notebooks:**
- `notebooks/babyai_pytorch_setup.ipynb` — Colab-ready end-to-end setup and training notebook (checks out `pytorch` branch + cherry-picks BabyAI files from `madison/babyai`)
- `notebooks/babyai.ipynb` — Development/inspection notebook for exploring the environment, viewing tool schemas, and visualizing gold trajectories

**External dependencies** (`gym`, `babyai_text`, `babyai`, `gym-minigrid`) are **not** in `pyproject.toml` and must be installed manually from [Grounding_LLMs_with_online_RL](https://github.com/flowersteam/Grounding_LLMs_with_online_RL). Key gotchas:
- These packages use the old `gym` (not `gymnasium`) 0.21-style API. Since BabyAI packages are installed with `--no-deps`, `gym` must be installed explicitly: `pip install "gym>=0.21,<0.27"`.
- They are incompatible with NumPy >= 2.0 (`np.bool8`, `np.int`, etc. removed). A compatibility shim in `env.py` patches `np.bool8`, but other aliases may need patching or you can pin `numpy<2`.
- See `src/act_prm/environments/babyai_text/README.md` for full installation instructions.

**Colab gotchas:**
- `!python` in notebook cells spawns a subprocess that does **not** inherit the kernel's `sys.path`. Set `PYTHONPATH` (include `src/` and BabyAI package paths) before running `main_pytorch.py` via `!python`.
- The `babyai_text` factory entry in `environments/__init__.py` only exists on `madison/babyai`, not on `pytorch`. The notebook patches it in at setup time. When patching, match the full `raise NotImplementedError(f"Sorry invalid environment: ...")` line — a partial match like `raise NotImplementedError` will hit the indented one inside the `act_prm_with_base_env` else block first (substring match bug).

## Environment Variables (`.env`)

```
TINKER_API_KEY="..."      # Required for Tinker backend
HF_TOKEN="..."            # HuggingFace API access
WANDB_API_KEY="..."       # Experiment tracking
WANDB_ENTITY="..."        # WandB team name
```
