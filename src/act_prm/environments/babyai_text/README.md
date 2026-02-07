# BabyAI-Text Environment

Adapter for the [BabyAI-Text](https://github.com/flowersteam/Grounding_LLMs_with_online_RL/tree/main/babyai-text) grid-world environment, wrapped into the act-prm `Environment` interface for use with the framework's generators, trainers, and replay buffers.

## Overview

BabyAI-Text is a text-observation variant of the [BabyAI](https://github.com/mila-iqia/babyai) grid-world platform. The agent receives text descriptions of its surroundings (instead of pixel images) and must complete missions like "pick up the red ball" or "go to the green door" by issuing discrete actions (turn left, turn right, go forward, pick up, drop, toggle).

This adapter:
- Wraps BabyAI-Text gym environments into the act-prm `Environment` interface
- Generates gold demonstration trajectories using the BabyAI `Bot` planner
- Formats observations and actions as chat-style messages for LLM training
- Supports both synchronous (`BabyAiTextEnv`) and async (`AsyncBabyAiTextEnv`) operation

## External Dependencies

The BabyAI packages are **not** listed in `pyproject.toml` and must be installed manually. You need three packages from the [Grounding_LLMs_with_online_RL](https://github.com/flowersteam/Grounding_LLMs_with_online_RL) repository:

| Package | Description |
|---|---|
| `babyai-text` | Text-observation wrapper for BabyAI; registers `BabyAI-*` environments with gym |
| `babyai` | Core BabyAI environment logic and the `Bot` planner |
| `gym-minigrid` | MiniGrid grid-world backend |

Additionally, you need these Python packages:
- `gym` (OpenAI Gym 0.21–0.26 — the old `gym`, **not** `gymnasium`). Since the BabyAI packages are installed with `--no-deps`, `gym` will **not** be pulled in automatically — you must install it explicitly: `pip install "gym>=0.21,<0.27"`
- `blosc`, `colorama`, `termcolor`, `matplotlib` (transitive dependencies of babyai)

## Installation

### 1. Clone the BabyAI repository

```bash
git clone https://github.com/flowersteam/Grounding_LLMs_with_online_RL.git
```

### 2. Install gym and supporting libraries

```bash
pip install "gym>=0.21,<0.27" blosc colorama termcolor matplotlib
```

**Note:** `gym` must be installed explicitly because `--no-deps` (step 3) prevents it from being pulled in as a transitive dependency. Make sure you install the old `gym` package, not `gymnasium`.

### 3. Install the three BabyAI packages (in order, editable mode, no deps)

```bash
cd Grounding_LLMs_with_online_RL/babyai-text
pip install -e . --no-deps

cd babyai
pip install -e . --no-deps

cd ../gym-minigrid
pip install -e . --no-deps
```

The `--no-deps` flag is important to avoid pulling in conflicting dependency versions.

### 4. Ensure packages are importable

If imports fail after installation (especially after a kernel restart in Colab), you may need to add the paths to `sys.path` manually:

```python
import sys
sys.path.insert(0, "/path/to/Grounding_LLMs_with_online_RL/babyai-text")
sys.path.insert(0, "/path/to/Grounding_LLMs_with_online_RL/babyai-text/babyai")
sys.path.insert(0, "/path/to/Grounding_LLMs_with_online_RL/babyai-text/gym-minigrid")
```

### 5. Verify installation

```python
import babyai_text      # Should register BabyAI-* envs with gym
import babyai            # Core BabyAI
import gym_minigrid      # MiniGrid backend
from babyai.bot import Bot  # Rule-based planner
```

### NumPy >= 2.0 Compatibility

The BabyAI packages (and their dependency `gym` 0.21/0.26) reference `np.bool8`, `np.int` and other NumPy type aliases that were removed in NumPy 2.0. This means **the BabyAI packages are incompatible with NumPy >= 2.0 out of the box**. If your environment has NumPy 2.x installed (common with recent PyTorch / HuggingFace stacks), you will see `AttributeError: module 'numpy' has no attribute 'bool8'` on import.

The adapter includes a compatibility shim in `env.py` that patches `np.bool8 = np.bool_` before importing gym/babyai, which resolves the most common crash. However, if you hit other removed aliases (`np.int`, `np.float`, `np.object`, etc.) deeper in gym or babyai, you have two options:

1. **Pin NumPy < 2.0** in the environment where BabyAI runs: `pip install "numpy<2"`
2. **Add more shims** at the top of your script before any gym/babyai imports:
   ```python
   import numpy as np
   if not hasattr(np, "bool8"):
       np.bool8 = np.bool_
   # Add others as needed:
   # np.int = int
   # np.float = float
   # np.object = object
   ```

The `--no-deps` install flag (step 3 above) helps avoid having pip downgrade or conflict with your existing NumPy, but be aware of this incompatibility when debugging import errors.

## Configuration

Environment configs live in `configs/environments/babyai/`:

| File | Split | Description |
|---|---|---|
| `default.yaml` | train | 1000 train / 64 val / 100 test samples, 20 max turns |
| `eval.yaml` | eval | Same sample counts, eval split |
| `test.yaml` | test | Same sample counts, test split |

The default gym environment is `BabyAI-MixedTestLocal-v0`. Other BabyAI-Text environments can be used by changing the `env_name` field in the config.

For Act-PRM training (wrapping BabyAI as a base env), see `configs/environments/act_prm/babyai.yaml`.

## Running

### Direct BabyAI training (RL or SFT)

```bash
uv run python main_pytorch.py \
  --env_config babyai/default \
  --eval_env_config babyai/eval \
  --generator_config default \
  --trainer_config <your_trainer_config> \
  --replay_buffer_config default \
  --model_config <your_model_config> \
  --lora_config <your_lora_config>
```

### Act-PRM over BabyAI

```bash
uv run python main_pytorch.py \
  --env_config act_prm/babyai \
  --base_env_config babyai/default \
  --eval_env_config babyai/eval \
  --generator_config aprm_qwen3_ap \
  --trainer_config aprm_for_sft100 \
  --replay_buffer_config default \
  --model_config <your_model_config> \
  --lora_config <your_lora_config>
```

## Colab Setup

A ready-to-run Colab notebook is available at `notebooks/babyai_pytorch_setup.ipynb`. It handles cloning, dependency installation (including the NumPy shim and explicit `gym` install), authentication, and training in one place.

## File Structure

```
babyai_text/
├── README.md        # This file
├── __init__.py      # Package exports
├── env.py           # Main environment: BabyAiTextEnv, AsyncBabyAiTextEnv
├── prompts.py       # System-level instruction prompt template
└── tools.py         # Action <-> tool-call name mapping and formatting
```
