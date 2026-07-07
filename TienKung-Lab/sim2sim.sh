#!/usr/bin/env bash
python legged_lab/scripts/sim2sim_h1_dwaq.py  \
    --checkpoint /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/logs/h1_dwaq/2026-07-06_20-40-13_payload_resume_m18000_upright25_angvel008_2/model_35600.pt  \
    --model /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/legged_lab/assets/h1_description/mjcf/scene_payload_horizontal.xml \
    --record-video
