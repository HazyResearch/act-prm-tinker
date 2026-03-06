#!/usr/bin/env bash
# TextWorld SFT training with intermittent online evaluation
#
# Usage:
#   ./scripts/run_tw_sft.sh <task> <difficulty> [ao|ao_hideobs] [gpu_id]
#
# Examples:
#   ./scripts/run_tw_sft.sh coin_collector easy ao 0
#   ./scripts/run_tw_sft.sh coin_collector easy ao_hideobs 0
#   ./scripts/run_tw_sft.sh treasure_hunter medium ao 1
#   ./scripts/run_tw_sft.sh treasure_hunter medium ao_hideobs 1
#
# Runs all 8 variants:
#   ./scripts/run_tw_sft_all.sh
set -euo pipefail

TASK="${1:?Usage: $0 <task> <difficulty> [ao|ao_hideobs] [gpu_id]}"
DIFFICULTY="${2:?Usage: $0 <task> <difficulty> [ao|ao_hideobs] [gpu_id]}"
VARIANT="${3:-ao}"        # "ao" or "ao_hideobs"
GPU="${4:-0}"

# --- Map task names to config short names ---
case "${TASK}" in
    coin_collector|coin) TASK_SHORT="coin" ;;
    treasure_hunter|treasure) TASK_SHORT="treasure" ;;
    *) echo "Unknown task: ${TASK}. Use coin_collector or treasure_hunter." && exit 1 ;;
esac

# --- Env config (SFT data) ---
ENV_CONFIG="act_lm/tw_${TASK_SHORT}_${DIFFICULTY}_${VARIANT}"

# --- Eval env config (online TextWorld) ---
case "${TASK_SHORT}" in
    coin) EVAL_TASK="coin_collector" ;;
    treasure) EVAL_TASK="treasure_hunter" ;;
esac
EVAL_ENV_CONFIG="textworld/${EVAL_TASK}_${DIFFICULTY}"

# --- Flags ---
EXTRA_FLAGS="--actions_only"
if [ "${VARIANT}" = "ao_hideobs" ]; then
    EXTRA_FLAGS="--actions_only --hide_observations"
fi

# --- Replicate tag ---
REPLICATE="sft_${VARIANT}"

echo "=== TextWorld SFT Training ==="
echo "  Task:       ${TASK} (${DIFFICULTY})"
echo "  Variant:    ${VARIANT}"
echo "  Env config: ${ENV_CONFIG}"
echo "  Eval config: ${EVAL_ENV_CONFIG}"
echo "  GPU:        ${GPU}"
echo "  Flags:      ${EXTRA_FLAGS}"
echo ""

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES="${GPU}" \
uv run python main_pytorch_lm.py \
  --env_config "${ENV_CONFIG}" \
  --eval_env_config "${EVAL_ENV_CONFIG}" \
  --model_config hf_qwen3_4b_inst_2507 \
  --lora_config r8_a16_qkvo \
  --trainer_config pt_sft_gen5 \
  --replay_buffer_config default \
  --log_path ./logs \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --seed 0 --replicate "${REPLICATE}" --verbose \
  ${EXTRA_FLAGS} \
  --learning_rate 1e-3 --mini_batch_size 8 --gradient_accumulation_steps 8 \
  --eval_every 5 --streamer --eval_rollout_every 10 --eval_rollout_start 200
