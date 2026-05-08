## Setup Environment

Create a conda environment:
```bash
conda env create -f conda_env.yml 
conda activate more
```
Install pytorch 2.3.1 with cuda-12.1:
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Download and install [Isaac Gym](https://developer.nvidia.com/isaac-gym):
```bash
cd isaacgym/python && pip install -e .
```
Install rsl_rl and legged gym:
```bash
cd rsl_rl && pip install -e . && cd .. 
pip install -e .
```

## Getting Started

1️⃣ **Train loco policy**: \
Train a locomotion policy with the stage-1 setup.
```bash
python legged_gym/scripts/train.py --task g1_16dof_loco --headless
```
* Train for 30k–50k iterations (recommended: ≥40k).
* Use at least 3000 environments for stable learning.

2️⃣ **Visualize**: \
After training, you can visualize the learned policy using the following command:
```bash
python legged_gym/scripts/play.py --task g1_16dof_loco --load_run ${policy_path}
```
 🕹️ Viewer Controls \
You can manually control the robot behaviors during visualization.

| Key | Function |
|:----:|:----------|
| `W, A, S, D` | Move forward, left, backward, right |
| `Z, X, C` | Switch gait command — `Z`: walk/run, `X`: high-knees, `C`: squat |
| `[ , ]` | Switch between robots |
| `Space` | Pause / Unpause simulation |
