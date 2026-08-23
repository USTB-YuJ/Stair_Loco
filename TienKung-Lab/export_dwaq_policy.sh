#!/usr/bin/env bash
set -euo pipefail

policy_export_dir="/root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/logs/h1_dwaq/2026-08-15_15-59-41_vertical_payload_feetfix_from25800/exported_policy"
mkdir -p "$policy_export_dir"

/opt/conda/envs/isaac/bin/python legged_lab/scripts/export_dwaq_policy.py \
        --checkpoint /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/logs/h1_dwaq/2026-08-15_15-59-41_vertical_payload_feetfix_from25800/model_35000.pt \
        --output "$policy_export_dir/policy.pt" \
        --num_obs 70 \
        --num_actions 19 \
        --history_length 5
