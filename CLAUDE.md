# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**act-prm-tinker** is a research codebase for training Action Process Reward Models (Act-PRMs) — LLMs trained to generate reasoning thoughts conditioned on observed actions, where the reward signal is P(action | thought, state). Two training backends are supported: Tinker (cloud LLM training service) and PyTorch/HuggingFace (local GPU training with PEFT/LoRA).

## Commands

Package manager is `uv`. Python >= 3.12.

```bash
# Install dependencies
uv sync

# Install FlashAttention-2 (optional, for GPU training)
uv pip install flash-attn --no-build-isolation-package flash-attn

# Tinker-based training (async, cloud)
CUDA_VISIBLE_DEVICES=0 uv run python main.py \
  --is_async --env_config act_prm/textworld_fs1 --eval_env_config textworld/treasure_hunter \
  --generator_config default --trainer_config qwen3_4b_aprm100_sft200_rl200 \
  --replay_buffer_config default --log_path ./logs \
  --model_name Qwen/Qwen3-4B-Instruct-2507 --lora_rank 32 --seed 42 --replicate 5

# PyTorch RL/Act-PRM training (local GPU)
uv run python main_pytorch.py \
  --env_config act_prm/snorkel_finance_fs1 --model_config hf_qwen3_4b_inst_2507 \
  --lora_config r16_a32_qkvo --generator_config aprm_qwen3_ap \
  --trainer_config aprm_for_sft100 --replay_buffer_config default \
  --log_path ./logs --actions_only --hide_observations \
  --batch_size 16 --group_size 8 --learning_rate 4e-5 --seed 42

# PyTorch SFT-only training (action language modeling)
uv run python main_pytorch_lm.py \
  --env_config act_lm/taubench_airline_gt --model_config hf_qwen3_4b_inst_2507 \
  --lora_config r8_a16_qkvo --trainer_config pt_sft_gen5 \
  --log_path ./logs --seed 0 --replicate gen1_sl_al

# Stage experiment metrics to git
uv run python stage_metrics.py --dry-run --verbose  # preview
uv run python stage_metrics.py --verbose             # actually stage
```

## Code Quality

- Run linter: `uv run ruff check .`
- Auto-fix issues: `uv run ruff check --fix .`
- Format code: `uv run ruff format .`
- GitHub Actions enforce linting on all pushes to main and PRs

Before committing, always run `ruff check .` and fix all issues. Code must be lint-clean.

## Shell Commands

Never use `$()` command substitution in shell commands. Instead, write the commit message to a temporary file and use `git commit -F`, or use `git commit -m` with a simple inline string. For multi-line messages, write to a file first with the Write tool, then `git commit -F /tmp/commit_msg.txt`.

## Git Workflow

Commit per logical task and push to `origin cc` immediately after.

**Commit message format**:
```
Brief summary of changes (imperative mood, <70 chars)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

**Branching rules**:
- Small fixes and additive changes: commit directly to `cc` and push.
- Potentially breaking feature additions: create a new branch, open a PR, merge after review.

When in doubt, prefer a branch + PR over a direct push to `cc`.
For new branches, create a new branch from `cc` and name them `cc/<branch-name>`. Also push to `origin cc/<branch-name>`, and setup a workflow that lets you make PRs and merge them into `cc`.

## Architecture

### Entry Points

- `main.py` — Tinker-based (cloud) async training loop
- `main_pytorch.py` — PyTorch RL/Act-PRM training loop (local GPU)
- `main_pytorch_lm.py` — PyTorch SFT-only training (supervised fine-tuning on demonstrations)

### Source Code (`src/act_prm/`)

**Environments** (`environments/`): RL-style environments with `reset()` → `EnvironmentState` and `step()` → `EnvironmentStepResult`. Registered via `get_env(name, is_async)` in `__init__.py`. Types: `act_prm`, `act_prm_with_base_env`, `action_lm`, `textworld`, `hotpotqa_mc`, `browsecomp_plus_search`, `longbench_v2`.

**Trainers**: Two parallel hierarchies:
- `trainer/tinker/` — Tinker-based async trainers: `act_prm`, `act_prm_sft_eval`, `act_prm_sft_rl`, `act_prm_joint`, `rl`, `sft`
- `pytorch/trainer/` — PyTorch local trainers: `act_prm_for_sft`, `rl`, `sft`
Both use `get_trainer(name)` factory in their respective `__init__.py`.

**Generators**: Handle rollout collection.
- `generator/` — Tinker-based: `TinkerGenerator`, `TinkerActPrmGenerator`, `TinkerActionPromptActPrmGenerator`
- `pytorch/generator/` — PyTorch-based: `HuggingFaceGenerator`, `ActionPromptActPrmGenerator`

**LLM Handlers** (`llm_handlers/`): Wrappers for model inference — `HuggingFaceLLM`, `OpenAIResponsesLLM`, `AsyncOpenAIResponsesLLM`, `TinkerCompleter`. Loaded via `load_llm()`.

**LoRA** (`lora/`): PEFT/LoRA utilities — `get_lora_model()`, `save_lora()`, `load_lora()`, adapter enable/disable helpers.

**Replay Buffer** (`replay_buffer/`): Stores rollout trajectories. Key types: `EpisodeStep` (Pydantic), `Trajectory`, `TrajectoryGroup`, `MeanCenteredTrajectoryGroup`.

**Utils** (`utils/`): `args.py` (CLI parser), `setup.py` (seeding, run name generation), `logging.py`, `display.py` (RichTextStreamer).

### Configuration System

OmegaConf-based hierarchical YAML configs in `configs/`. CLI args override config values.

| Directory | Purpose |
|---|---|
| `configs/environments/act_prm/` | Act-PRM env configs |
| `configs/environments/act_lm/` | Action LM (SFT) env configs |
| `configs/environments/textworld/` | TextWorld game configs |
| `configs/environments/hotpotqa_mc/` | HotpotQA configs |
| `configs/environments/browsecomp_plus/` | BrowseComp configs |
| `configs/generator/` | Generator/sampling configs |
| `configs/trainer/` | Trainer hyperparameter configs |
| `configs/model/` | HuggingFace model configs |
| `configs/lora/` | LoRA rank/alpha/target configs |

Run names are auto-generated from CLI args as abbreviated key=value pairs. Logs go to `<log_path>/<env_config_name>/<model_name>/<run_name>/`.

### Environment Variables

Set via `.env` file in project root (loaded by `python-dotenv`):
- `TINKER_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, `WANDB_ENTITY`

### Key Concepts

- **Act-PRM**: Train LLM to predict P(action | thought, state) — thought is reasoning, action is tool call/output
- **Action LM**: SFT mode — train directly on agent demonstration datasets without RL
- **Hide observations**: Redact past tool observations (replaced with `"..."`) to reduce context length
- **Group rollouts**: Generate `group_size` trajectories per task for variance reduction (GRPO-style)
