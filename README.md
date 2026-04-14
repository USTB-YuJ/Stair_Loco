# G1 DWAQ 盲走上台阶

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-2.3.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](LICENSE)

## 项目简介

本项目展示了基于深度强化学习的 **G1 29自由度人形机器人盲走上台阶** 的完整流程，包括：

- 🏋️ **仿真训练**：使用 IsaacLab 在 Nvidia Isaac Sim 中训练策略
- 🔄 **Sim2Sim 转移**：从 IsaacLab 仿真环境迁移到 MuJoCo 
- 🤖 **实物部署**：在 Unitree G1 真实机器人上部署执行

本项目在Isaaclab复现 **DreamWaQ** 算法，DreamWaQ算法部分参考[Manaro-Alpha](https://github.com/Manaro-Alpha/DreamWaQ)，框架基于天工[TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)和[Legged Lab](https://github.com/Hellod035/LeggedLab)开源框架，实物部署由[LeggedLabDeploy](https://github.com/Hellod035/LeggedLabDeploy)实现。

## 项目结构

```
G1DWAQ_Lab/
├── IsaacLab/                  # Isaac Lab 框架和仿真环境
├── TienKung-Lab/              # 基于 Legged Lab 的训练代码（含 DWAQ 环境定义）
├── LeggedLabDeploy/           # 实物部署代码（支持 DWAQ 配置）
├── unitree_sdk2_python/       # Unitree 机器人通信 SDK
├── LICENSE                    # 项目许可证
└── README.md                  # 本文件
```

## 快速开始

### 环境配置

#### 1. 安装 Isaac Lab

请按照 [Isaac Lab 官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装。建议使用 conda 环境便于从终端调用 Python 脚本。

#### 2. 获取项目代码

```bash
# 克隆本项目
git clone https://github.com/liuyufei-nubot/G1DWAQ_Lab.git
cd G1DWAQ_Lab
```

#### 3. 安装依赖

```bash
# 安装 TienKung-Lab
cd TienKung-Lab
pip install -e .

# 安装 rsl_rl
cd TienKung-Lab/rsl_rl
pip install -e .

# 安装 Unitree SDK (用于实物部署)
cd ../unitree_sdk2_python
pip install -e .

# 返回项目根目录
cd ..
```

### 训练

使用 DWAQ 算法训练 G1 机器人盲走上台阶（仅在RTX4090单卡测试）：

```bash
cd TienKung-Lab

# 训练
python legged_lab/scripts/train.py --task=g1_dwaq --headless --num_envs=4096 --max_iterations=10000
```

**参数说明**：
- `--task`: 任务名称（`g1_dwaq`, `g1_rough` 等）
- `--headless`: 无图形界面运行（推荐用于训练）
- `--num_envs`: 并行环境数量（根据 GPU 显存调整）
- `--max_iterations`: 最大训练迭代数

**训练输出**：
- 日志文件位置：`logs/g1_dwaq/<时间戳>/`
- 模型检查点：`logs/g1_dwaq/<时间戳>/model_<迭代数>.pt`

### 测试/推断

#### 仿真测试

```bash
cd TienKung-Lab

python legged_lab/scripts/play.py --task=g1_dwaq --load_run=<运行目录> --checkpoint=model_<迭代次数>.pt
```

示例：
```bash
python legged_lab/scripts/play.py --task=g1_dwaq --load_run=2026-01-16_00-46-00 --checkpoint=model_9999.pt
```

#### Sim2Sim 转移

实物部署前推荐Sim2Sim测试：

```bash
python legged_lab/scripts/sim2sim_g1_dwaq.py --scene stairs
```

### 实物部署

详细的实物部署指南请参考 [LeggedLabDeploy/README_DWAQ.md](LeggedLabDeploy/README_DWAQ.md)

#### 快速部署流程

1. **导出策略**：
   ```bash
   cd TienKung-Lab
   python legged_lab/scripts/export_dwaq_policy.py \
       --checkpoint logs/g1_dwaq/2026-01-15_11-21-04/model_9999.pt
   ```

2. **复制策略文件**：
   ```bash
   mkdir -p ../LeggedLabDeploy/policy/g1_dwaq
   cp logs/g1_dwaq/2026-01-15_11-21-04/exported/policy.pt \
      ../LeggedLabDeploy/policy/g1_dwaq/
   ```

3. **启动部署**：
   ```bash
   cd ../LeggedLabDeploy
   python deploy.py --config_path configs/g1_dwaq_jit.yaml --net <网卡名称>
   ```

## 核心技术细节

### G1 机器人配置

**自由度配置**：29 DOF

当前训练使用的是 **Unitree G1 29自由度版本**，基于 `g1_29dof_simple_collision.urdf` 模型。

### 观测空间

| 观测项 | 维度 |
|--------|------|
| 角速度 (body frame) | 3 |
| 重力投影 (body frame) | 3 |
| 速度命令 [vx, vy, yaw_rate] | 3 |
| 关节位置偏差 | 29 |
| 关节速度 | 29 |
| 上一步动作 | 29 |
| 步态相位 (可选) | 1 |
| **总计** | **96 / 100** |

### 两个模型版本

| 版本 | 维度 | 特点 | 推荐场景 |
|------|------|------|---------|
| 无步态版本 (`g1_dwaq_jit.yaml`) | 96 | 原地转弯困难 | 台阶、复杂地形 |
| 带步态版本 (`g1_dwaq_phase.yaml`) | 100 | 转向能力强 (+42%) | 台阶、复杂地形 |

## 功能特性

### ✅ 已实现

- [x] DWAQ 算法基础实现
- [x] IsaacLab 仿真环境
- [x] 强化学习训练流程
- [x] Sim2Sim 转移脚本
- [x] 实物部署支持
- [x] 多环境并行训练
- [x] TorchScript 模型导出
- [x] 无步态和带步态两个版本

### 📋 可扩展方向

- [ ] 视觉输入集成
- [ ] 显式地形估计

## 常见问题排查

### TODO......

## 贡献指南

欢迎提交 Issue 和 Pull Request！

- 添加适当的测试用例
- 更新相关文档

## 致谢

本项目基于以下优秀开源项目：

- **[Legged Lab](https://github.com/Hellod035/LeggedLab)** - 提供了直接、透明的 IsaacLab 工作流，以及可复用的强化学习组件。Legged Lab 的代码组织和环境定义可无缝从 IsaacGym 迁移到IsaacLab。

- **[天工开源框架 (TienKung-Lab)](https://github.com/Open-X-Humanoid/TienKung-Lab)** - 开源框架提供了高质量的足式机器人学习环境实现和最佳实践。

- **[IsaacLab](https://github.com/isaac-sim/IsaacLab)** - NVIDIA 官方的 Isaac Lab 提供了强大的仿真和强化学习工具。

- **[RSL_RL](https://github.com/leggedrobotics/rsl_rl)** - 提供了高效的强化学习算法实现。

- **[Unitree Robotics](https://github.com/unitreerobotics)** - 提供了 G1 机器人的硬件接口和 SDK。

## 引用

如果在您的研究中使用了本项目，请使用以下方式引用：

```bibtex
@software{G1DWAQBlindStairs,
  title = {G1 DWAQ: Blind Stair Climbing for Unitree G1 Humanoid Robot},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/your-username/G1DWAQ_Lab},
  license = {BSD-3-Clause}
}
```

## 许可证

本项目采用 [BSD-3-Clause License](LICENSE) 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 GitHub Issues
- 发送邮件: yufei.liu@nudt.edu.cn
- 参与讨论和贡献
