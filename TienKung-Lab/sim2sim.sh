#!/usr/bin/env bash
python legged_lab/scripts/sim2sim_h1_dwaq.py  \
    --checkpoint /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/logs/h1_dwaq/2026-08-15_15-59-41_vertical_payload_feetfix_from25800/model_35000.pt  \
    --model /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/legged_lab/assets/h1_description/mjcf/scene_payload_vertical_3step.xml \
    "$@"
