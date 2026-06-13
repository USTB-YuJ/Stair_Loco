#!/usr/bin/env bash
set -e

PYTHON=${PYTHON:-/opt/conda/envs/isaac/bin/python}
"${PYTHON}" legged_lab/scripts/sim2sim_h1_dwaq.py   --checkpoint /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/logs/h1_dwaq/2026-06-12_11-47-51_resume_dwaq_low_level/model_18000.pt   --model /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/legged_lab/assets/h1_description/mjcf/scene.xml   --record-video
