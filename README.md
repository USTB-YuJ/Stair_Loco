# H1 DWAQ 用户手册

[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://docs.python.org/3/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](LICENSE)

## 项目简介

本仓库当前面向 **Unitree H1 人形机器人**，提供一条完整的 H1 DWAQ 使用链路：

- **训练**：在 `TienKung-Lab` 中训练 `h1_dwaq` 策略
- **仿真验证**：在 Isaac Sim 中 `play`，并在 MuJoCo 中进行 `sim2sim`
- **策略导出**：将 H1 DWAQ checkpoint 导出为 TorchScript
- **实机部署**：在 `LeggedLabDeploy` 中使用 `deploy.py` 部署到 H1 实机

这里的 DWAQ 策略是当前仓库主推的 H1 盲行走/带步态相位控制方案。  
如果你是第一次使用本仓库，建议直接按本文的 `train -> play -> sim2sim -> export -> deploy` 顺序操作。

## 仓库结构

```text
G1DWAQ_Lab-main/
├── TienKung-Lab/        # H1 DWAQ 训练、play、sim2sim、导出脚本
├── LeggedLabDeploy/     # H1 实机部署代码与配置
├── unitree_sdk2_python/ # Unitree 通信 SDK
├── LICENSE
└── README.md
```

### 两个子工程分别负责什么

- `TienKung-Lab`
  - `h1_dwaq` 任务定义
  - Isaac Sim 训练和 play
  - MuJoCo sim2sim
  - DWAQ policy 导出

- `LeggedLabDeploy`
  - H1 实机控制主程序 `deploy.py`
  - H1 部署配置 `configs/h1.yaml`
  - H1 DWAQ 部署配置 `configs/h1_dwaq_phase.yaml`

## 快速开始

## 1. 环境准备

建议使用已经安装好 Isaac Lab 的 Python/conda 环境。

### 安装训练侧依赖

```bash
cd TienKung-Lab
pip install -e .

cd rsl_rl
pip install -e .
```

### 安装部署侧依赖

```bash
cd ../../unitree_sdk2_python
pip install -e .
```

如果你主要做训练和 sim2sim，可以先只安装 `TienKung-Lab` 和 `rsl_rl`。  
只有在需要上机部署时，才必须安装 `unitree_sdk2_python`。

## 2. H1 DWAQ 训练

仓库里已经给了一个 H1 DWAQ 训练脚本：

```bash
cd TienKung-Lab
bash train.sh
```

其对应的实际命令是：

```bash
python legged_lab/scripts/train.py \
    --task=h1_dwaq \
    --gpu="0" \
    --logger=tensorboard \
    --num_envs=4096 \
    --max_iterations=80000 \
    --headless
```

训练输出默认位于：

```text
TienKung-Lab/logs/h1_dwaq/<run_timestamp>/
```

其中最重要的是：

- `model_<iter>.pt`：checkpoint
- `params/env.yaml`：环境配置快照
- `params/agent.yaml`：算法配置快照

## 3. Isaac Sim 中回放策略

训练完成后，先在仿真中检查策略是否正常：

```bash
cd TienKung-Lab
bash play.sh
```

当前脚本内容等价于：

```bash
python legged_lab/scripts/play.py \
    --task=h1_dwaq \
    --num_envs=10 \
    --load_run=2026-04-17_10-22-42 \
    --checkpoint=model_8000.pt \
    --difficulty=0.1
```

建议你根据自己的 run 目录改这几个参数：

- `--load_run`
- `--checkpoint`
- `--difficulty`

## 4. MuJoCo sim2sim 验证

上机前，建议先做 sim2sim：

```bash
cd TienKung-Lab
bash sim2sim.sh
```

当前脚本等价于：

```bash
python legged_lab/scripts/sim2sim_h1_dwaq.py \
  --checkpoint /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/logs/h1_dwaq/2026-04-17_10-22-42/model_9000.pt \
  --model /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab/legged_lab/assets/h1_description/mjcf/scene.xml
```

推荐把 sim2sim 作为上机前的必经环节。  
如果这里已经出现明显的不稳定、姿态异常、NaN 或关节爆炸，就不要直接上实机。

## 5. 导出 H1 DWAQ TorchScript

H1 DWAQ 部署前，需要先把 checkpoint 导出为 TorchScript。

可以直接使用仓库中的导出脚本：

```bash
cd TienKung-Lab
bash export_dwaq_policy.sh
```

当前脚本等价于：

```bash
python legged_lab/scripts/export_dwaq_policy.py \
    --checkpoint logs/h1_dwaq/2026-04-17_10-22-42/model_5300.pt \
    --num_obs 70 \
    --num_actions 19 \
    --history_length 5
```

导出完成后，通常会得到：

```text
TienKung-Lab/logs/h1_dwaq/<run_timestamp>/exported/policy.pt
```

## 6. H1 实机部署

H1 DWAQ 的实机部署已经整理为单独文档：

- [LeggedLabDeploy/README_H1_DWAQ.md](LeggedLabDeploy/README_H1_DWAQ.md)

最简流程如下：

### 复制策略到部署目录

```bash
cd /root/gpufree-data/workspace/G1DWAQ_Lab-main
mkdir -p LeggedLabDeploy/policy/h1_dwaq_phase
cp TienKung-Lab/logs/h1_dwaq/<run_timestamp>/exported/policy.pt \
   LeggedLabDeploy/policy/h1_dwaq_phase/
```

### 使用 H1 DWAQ 专用配置启动

```bash
cd LeggedLabDeploy
python deploy.py --config_path configs/h1_dwaq_phase.yaml --net <network_interface>
```

其中：

- `configs/h1_dwaq_phase.yaml` 是 H1 带步态 DWAQ 的部署配置
- `<network_interface>` 是连接 H1 机器人的网卡名，例如 `eno1`

## H1 DWAQ 关键配置

当前 H1 DWAQ 主线有 4 个最重要的配置事实：

1. **动作维度**：`19`
2. **单帧观测维度**：`70`
3. **历史长度**：`5`
4. **步态相位**：启用

### 70 维观测组成

```text
ang_vel (3)
+ projected_gravity (3)
+ command (3)
+ joint_pos (19)
+ joint_vel (19)
+ previous_action (19)
+ gait_phase (4)
= 70
```

### gait phase 参数

```text
period = 0.8
offset = 0.5
```

### gait phase 顺序

部署侧必须与训练侧保持一致：

```text
[sin_left, sin_right, cos_left, cos_right]
```

这已经在 `LeggedLabDeploy/deploy.py` 中按 H1 DWAQ 的训练观测顺序处理好了。

## 推荐使用顺序

如果你想快速跑通整条 H1 工作流，建议按这个顺序：

1. `bash train.sh`
2. `bash play.sh`
3. `bash sim2sim.sh`
4. `bash export_dwaq_policy.sh`
5. 按 `LeggedLabDeploy/README_H1_DWAQ.md` 上机部署

## 离线验证建议

如果暂时不上机，也建议在部署前至少完成这三步：

1. **play 验证**  
   检查 H1 DWAQ 在 Isaac Sim 中能否稳定站立和行走。

2. **sim2sim 验证**  
   检查策略是否能在 MuJoCo 中保持稳定。

3. **导出验证**  
   确认 `policy.pt` 可以正常导出并用于部署链路。

如果要更严格一些，建议再做：

- checkpoint 与导出 TorchScript 的动作对比
- `deploy.py` 的离线回放验证

## 常见入口文件

### 训练侧

- `TienKung-Lab/legged_lab/envs/h1/h1_dwaq_config.py`
- `TienKung-Lab/legged_lab/scripts/train.py`
- `TienKung-Lab/legged_lab/scripts/play.py`
- `TienKung-Lab/legged_lab/scripts/sim2sim_h1_dwaq.py`
- `TienKung-Lab/legged_lab/scripts/export_dwaq_policy.py`

### 部署侧

- `LeggedLabDeploy/deploy.py`
- `LeggedLabDeploy/configs/h1.yaml`
- `LeggedLabDeploy/configs/h1_dwaq_phase.yaml`
- `LeggedLabDeploy/README_H1_DWAQ.md`

## 常见问题

### 1. 为什么不能直接使用 `configs/h1.yaml` 部署 H1 DWAQ？

因为 `configs/h1.yaml` 对应的是普通 H1 策略，不是 DWAQ：

- `history_length = 10`
- `num_obs = 66`

而 H1 带步态 DWAQ 部署需要：

- `history_length = 5`
- `num_obs = 70`

### 2. 为什么导出时必须显式传 `--num_obs 70 --num_actions 19 --history_length 5`？

因为 `export_dwaq_policy.py` 的默认值更偏向另一套配置。  
如果不显式覆盖，导出的 TorchScript 结构可能和 H1 DWAQ 不匹配。

### 3. 上机前最少要做哪些验证？

至少做：

- `play`
- `sim2sim`
- TorchScript 导出检查

不要在 checkpoint 还没经过这三层验证时直接上 H1 实机。

## 安全提示

H1 实机部署存在风险，首次测试时务必注意：

1. 使用吊架或保护装置
2. 确保周围有足够空间
3. 优先从小速度命令开始
4. 随时准备按 `select` 紧急停止
5. 如果 sim2sim 还不稳定，就不要上机

## 参考文档

- 训练框架说明：[`TienKung-Lab/README.md`](TienKung-Lab/README.md)
- H1 DWAQ 实机部署：[`LeggedLabDeploy/README_H1_DWAQ.md`](LeggedLabDeploy/README_H1_DWAQ.md)

## 致谢

本仓库基于以下项目构建和扩展：

- [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)
- [Legged Lab](https://github.com/Hellod035/LeggedLab)
- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl)
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)

## 许可证

本项目采用 [BSD-3-Clause License](LICENSE)。
