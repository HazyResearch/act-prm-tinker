#!/usr/bin/env bash
# Run all 8 TextWorld SFT training variants sequentially on a single GPU.
# For parallel runs, call run_tw_sft.sh directly with different GPU IDs.
#
# Usage:
#   ./scripts/run_tw_sft_all.sh [gpu_id]
set -euo pipefail

GPU="${1:-0}"
SCRIPT_DIR="$(dirname "$0")"

TASKS=("coin_collector" "treasure_hunter")
DIFFICULTIES=("easy" "medium")
VARIANTS=("ao" "ao_hideobs")

for task in "${TASKS[@]}"; do
    for diff in "${DIFFICULTIES[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            echo ""
            echo "================================================================"
            echo "  ${task} / ${diff} / ${variant}"
            echo "================================================================"
            bash "${SCRIPT_DIR}/run_tw_sft.sh" "${task}" "${diff}" "${variant}" "${GPU}"
        done
    done
done

echo ""
echo "All 8 variants completed."
