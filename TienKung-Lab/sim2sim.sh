#!/usr/bin/env bash
python legged_lab/scripts/sim2sim_h1_dwaq.py  \
    --checkpoint /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/logs/h1_dwaq/2026-06-28_15-59-06_payload_resume_m18000_upright25_angvel008/model_25800.pt  \
    --model /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/legged_lab/assets/h1_description/mjcf/scene_payload_horizontal.xml \
    --record-video
