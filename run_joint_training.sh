#!/bin/bash
# Joint Act-PRM training: gen_think + think_act on a single LLM
#
# Usage:
#   bash run_joint_training.sh
#
# Requires:
#   - act_prm env config (e.g., act_prm/tw_coin_easy_sp)
#   - textworld eval env config (e.g., textworld/coin_collector_easy)
#   - generator config for action-prompted generation (e.g., aprm_qwen3_ap)

set -euo pipefail
cd /scr/mzhang/projects/act-prm-tinker

# --- Configuration ---
PROJECT_NAME="act-prm-cc"
ENV_CONFIG="act_prm/tw_coin_easy_sp"
EVAL_ENV_CONFIG="textworld/coin_collector_easy"
MODEL_CONFIG="hf_qwen3_4b_inst_2507"
LORA_CONFIG="r16_a32_qkvo"
GENERATOR_CONFIG="aprm_qwen3_ap"
TRAINER_CONFIG="pt_aprm_joint100"
REPLAY_BUFFER_CONFIG="default"

BATCH_SIZE=16
GROUP_SIZE=8
LEARNING_RATE=4e-5
NUM_SUBSTEPS=4
SEED=42
REPLICATE=0

# --- Run ---
uv run python main_pytorch.py \
    --project_name "$PROJECT_NAME" \
    --env_config "$ENV_CONFIG" \
    --eval_env_config "$EVAL_ENV_CONFIG" \
    --model_config "$MODEL_CONFIG" \
    --lora_config "$LORA_CONFIG" \
    --generator_config "$GENERATOR_CONFIG" \
    --trainer_config "$TRAINER_CONFIG" \
    --replay_buffer_config "$REPLAY_BUFFER_CONFIG" \
    --log_path ./logs \
    --actions_only --hide_observations \
    --batch_size "$BATCH_SIZE" \
    --group_size "$GROUP_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_substeps "$NUM_SUBSTEPS" \
    --seed "$SEED" --replicate "$REPLICATE" \
    --verbose
