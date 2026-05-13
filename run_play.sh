#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH=/root/gpufree-data/conda_space/envs/more/lib:$LD_LIBRARY_PATH
export PATH=/root/gpufree-data/conda_space/envs/more/bin:$PATH
exec python "$SCRIPT_DIR/legged_gym/scripts/play.py" "$@"
