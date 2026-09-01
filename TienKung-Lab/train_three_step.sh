#!/usr/bin/env bash
set -euo pipefail

export WARP_CACHE_PATH=/tmp/warp_h1_dwaq_three_step

BASE_RUN=2026-08-15_15-59-41_vertical_payload_feetfix_from25800
BASE_LOG_ROOT=logs/h1_dwaq_three_step
mkdir -p "$BASE_LOG_ROOT"
if [ ! -e "$BASE_LOG_ROOT/$BASE_RUN" ]; then
    ln -s "../h1_dwaq/$BASE_RUN" "$BASE_LOG_ROOT/$BASE_RUN"
fi

python legged_lab/scripts/train.py \
    --task=h1_dwaq_three_step \
    --gpu=0 \
    --logger=tensorboard \
    --num_envs=4096 \
    --max_iterations=50000 \
    --headless \
    --resume=True \
    --reset_optimizer \
    --freeze_dwaq_context \
    --load_run="$BASE_RUN" \
    --checkpoint=model_42600.pt \
    --run_name=three_step_corridor_transfer_freeze_context
