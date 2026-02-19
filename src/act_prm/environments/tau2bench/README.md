# tau2-bench Online Environment

Online environment for [tau2-bench](https://github.com/sierra-research/tau2-bench) tasks. Runs step-by-step rollouts with an LLM-powered simulated user, enabling rollout evaluation and RL training.

## Setup

### 1. Install tau2

```bash
uv pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench.git"
```

Or add to `pyproject.toml` (already included):
```toml
"tau2 @ git+https://github.com/sierra-research/tau2-bench.git",
```

### 2. Set API Keys

The simulated user requires an OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Add to your `.env` file at project root alongside other keys.

## Architecture

### How It Works

1. **Wraps `AgentGymEnv`**: tau2 provides a Gymnasium-compatible environment that handles all orchestration (agent ↔ user ↔ tools message routing, user simulation, tool execution, evaluation) in a background thread.

2. **`respond_user` tool**: A synthetic tool (consistent with offline training format from `action_lm/env_utils/tau_bench.py`) ensures the agent always produces structured tool calls, even for user-facing messages.

3. **Per-episode fresh env**: Each `reset()` creates a new `AgentGymEnv` for the selected task. Each `step()` routes parsed `ActionFromLLM` objects through tau2's orchestrator.

### Integration with Training

- **Environments**: `Tau2BenchEnv` (sync) and `AsyncTau2BenchEnv` (async with `asyncio.to_thread`)
- **Configs**: `configs/environments/tau2bench/airline.yaml`, `configs/environments/tau2bench/retail.yaml`
- **Factory**: Registered in `environments/__init__.py` as `"tau2bench"`

## Usage

### Configs

- `configs/environments/tau2bench/airline.yaml`
- `configs/environments/tau2bench/retail.yaml`

Key settings:
- `domain`: tau2 domain (`"airline"`, `"retail"`, `"telecom"`)
- `user_llm`: LLM for the simulated user (default: `"gpt-4.1-2025-04-14"`)
- `max_steps`: Max steps in tau2's orchestrator per episode
- `max_turns`: Max LLM turns before our-level truncation
- `num_train_tasks` / `num_test_tasks`: Train/test split sizes

### Running Rollout Evaluations

**Tinker-based:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python main.py \
  --is_async --env_config tau2bench/airline \
  --generator_config aprm_qwen3_ap \
  --trainer_config qwen3_4b_aprm \
  --replay_buffer_config default \
  --log_path ./logs \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --lora_rank 32 \
  --seed 42 --replicate 0 --verbose
```

**PyTorch-based:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python main_pytorch.py \
  --env_config tau2bench/airline \
  --model_config hf_qwen3_4b_inst_2507 \
  --lora_config r16_a32_qkvo \
  --generator_config aprm_qwen3_ap \
  --trainer_config aprm_for_sft100 \
  --replay_buffer_config default \
  --log_path ./logs \
  --seed 42 --replicate 0 --verbose
```

### Programmatic Usage

```python
from act_prm.environments import get_env

env = get_env(
    "tau2bench",
    domain="airline",
    user_llm="gpt-4.1-2025-04-14",
    num_train_tasks=5,
    num_test_tasks=2,
)
print(f"Number of train tasks: {len(env)}")

# Reset for a specific task
state = env.reset(sample_idx=0, generation_idx=0)
print(f"Tools: {[t['name'] for t in state.tools]}")
print(f"Initial message: {state.new_messages[0]['content']}")

# Step with a respond_user action
from act_prm.llm_handlers.types import ActionFromLLM

action = ActionFromLLM(
    role="assistant",
    type="function_call",
    name="respond_user",
    arguments={"text": "Hello! How can I help you today?"},
    call_id="call_0",
)
result = env.step(
    parsed_actions=[action],
    model_response=None,
    current_state=state,
    current_messages=[],
)
print(f"Done: {result.done}, Reward: {result.reward}")
```

## Supported Domains

| Domain | Description |
|---|---|
| `airline` | Airline customer service (booking, cancellation, rebooking) |
| `retail` | Retail customer service (orders, returns, exchanges) |
| `telecom` | Telecom customer service (plans, billing, support) |
