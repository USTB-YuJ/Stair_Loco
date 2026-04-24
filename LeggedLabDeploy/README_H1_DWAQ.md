# H1 带步态 DWAQ 实机部署指南

## 目标

本文档说明如何把 `TienKung-Lab` 中训练得到的 **H1 带步态 DWAQ 策略** 部署到实机。

- 训练侧导出 TorchScript
- 部署侧使用 `deploy.py`

这是当前最推荐的方式，因为它直接复用现有实机控制框架，不需要在部署端重建 DWAQ 网络。

---

## 结论先看

### 可以直接复用的部分

`deploy.py` 已经具备部署 H1 DWAQ 所需的核心能力：

1. 可以加载 TorchScript 策略
2. 可以维护固定长度的观测历史
3. 已经支持可选的 gait phase 观测
4. `configs/h1.yaml` 里的 H1 电机映射、PD 增益、默认站立位姿都可以继续复用

### 需要修改/新增的部分

为了正确部署 **H1 带步态 DWAQ**，需要满足下面三点：

1. **不能直接使用 `configs/h1.yaml`**
   - `h1.yaml` 对应的是普通 H1 策略，不是 DWAQ
   - 它的 `history_length=10`、`num_obs=66`
   - 而当前 H1 下肢专训 DWAQ 需要的是 `history_length=5`、`num_obs=40`

2. **需要使用 DWAQ 专用导出脚本**
   - `export_dwaq_policy.py` 会把 `Encoder + Actor` 合成一个单输入 TorchScript
   - 部署时输入是 `5 * 40 = 200` 维的扁平化观测历史

3. **步态相位顺序必须与训练侧一致**
   - 训练侧真实顺序是：
     - `[sin_left, sin_right, cos_left, cos_right]`
   - 不是：
     - `[sin_left, cos_left, sin_right, cos_right]`
   - 当前仓库已经同步修正 `deploy.py` 为训练一致的顺序

---

## 当前 H1 DWAQ 的训练侧假设

根据 `TienKung-Lab/legged_lab/envs/h1/h1_dwaq_config.py`、`TienKung-Lab/legged_lab/envs/g1/g1_dwaq_config.py` 和 `TienKung-Lab/legged_lab/scripts/sim2sim_h1_dwaq.py`，H1 带步态 DWAQ 的关键部署假设如下：

- 动作维度：`10`（只控制下肢）
- 单帧 actor 观测维度：`40`
- DWAQ 历史长度：`5`
- gait phase：开启
- gait phase 参数：
  - `period = 0.8`
  - `offset = 0.5`
- 固定关节：腰部和上肢保持默认位姿

### 40 维观测组成

```text
ang_vel (3)
+ projected_gravity (3)
+ command (3)
+ joint_pos (10)
+ joint_vel (10)
+ previous_action (10)
+ gait_phase (4)
= 40
```

### gait phase 观测顺序

```text
[sin_left, sin_right, cos_left, cos_right]
```

这点非常重要。部署侧如果顺序不一致，策略即使能运行，步态和转向也会明显异常。

---

## 推荐部署方式

### 方式 A：`deploy.py` + TorchScript（推荐）

优点：

- 与当前 `LeggedLabDeploy` 的实机流程完全一致
- 不需要在部署端加载 DWAQ checkpoint 并重建模型
- 链路更短，出错面更小

本文档后续只详细说明这种方式。
---

## 仓库中已补充的内容

为了让 H1 带步态 DWAQ 能直接按 `deploy.py` 使用，当前工程已补充：

1. `configs/h1_dwaq_phase.yaml`
   - H1 带步态 DWAQ 专用部署配置

2. `deploy.py`
   - gait phase 顺序已修正为训练一致：
   - `[sin_left, sin_right, cos_left, cos_right]`

---

## 部署步骤

## 1. 先确认训练模型是否为 H1 带步态 DWAQ

你要部署的 checkpoint 需要同时满足：

- 任务是 `h1_dwaq`
- 观测带 gait phase
- 单帧观测维度为 `40`
- 历史长度为 `5`
- 动作维度为 `10`

如果你的模型不是这组维度，就不要直接套用本文的配置文件。

---

## 2. 从训练侧导出 TorchScript

进入训练工程：

```bash
cd ~/code/geoloco/TienKung-Lab
conda activate geo
```

执行导出：

```bash
python legged_lab/scripts/export_dwaq_policy.py \
    --checkpoint logs/h1_dwaq/<run_name>/model_<iter>.pt \
   --num_obs 40 \
   --num_actions 10 \
    --history_length 5
```

例如：

```bash
python legged_lab/scripts/export_dwaq_policy.py \
    --checkpoint logs/h1_dwaq/2026-xx-xx_xx-xx-xx/model_10000.pt \
   --num_obs 40 \
   --num_actions 10 \
    --history_length 5
```

导出后通常得到：

```text
logs/h1_dwaq/<run_name>/exported/policy.pt
```

### 这里为什么一定要显式传参

因为 `export_dwaq_policy.py` 的默认值是 G1 DWAQ：

- `num_obs=96`
- `num_actions=29`
- `history_length=5`

如果你不覆盖这些默认值，导出的模型结构就会和 H1 DWAQ 不匹配。

---

## 3. 复制策略到部署工程

进入部署工程根目录后，创建 H1 DWAQ 策略目录：

```bash
cd /LeggedLabDeploy
mkdir -p policy/h1_dwaq_phase
```

复制导出的 TorchScript：

最终路径应为：

```text
LeggedLabDeploy/policy/h1_dwaq_phase/policy.pt
```

---

## 4. 使用 H1 DWAQ 专用配置

部署时使用新配置文件：

```text
configs/h1_dwaq_phase.yaml
```

它和普通 `configs/h1.yaml` 的主要区别如下：

| 项目 | `h1.yaml` | `h1_dwaq_phase.yaml` |
|------|-----------|----------------------|
| 策略类型 | 普通 H1 策略 | H1 带步态 DWAQ |
| `policy_path` | `policy/h1/policy.pt` | `policy/h1_dwaq_phase/policy.pt` |
| `history_length` | 10 | 5 |
| `num_obs` | 66 | 40 |
| `num_actions` | 19 | 10 |
| `gait_phase.enable` | 无 | `true` |

### 说明

- `joint2motor_idx`
- `kps`
- `kds`
- `default_joint_pos`
- `msg_type`
- `imu_type`

这些 H1 机器人相关配置，都是沿用现有 `h1.yaml` 的，不需要重新发明一套。

---

## 5. 启动机器人并运行部署

确保：

- H1 已开机
- 遥控器可用
- 电脑通过网线连接机器人
- 已安装 `unitree_sdk2_python`

查看网卡名：

```bash
ip link show
```

假设连接机器人的网卡为 `eno1`，则执行：

```bash
cd /LeggedLabDeploy
python deploy.py --config_path configs/h1_dwaq_phase.yaml --net eno1
```

---

## 6. 实机控制流程

和当前 `deploy.py` 的标准流程一致：

1. 程序启动后进入零力矩状态
2. 按遥控器 `start`
   - 机器人缓慢移动到默认站立位姿
3. 确认站姿正常后，按 `A`
   - 进入策略控制
4. 摇杆控制：
   - 左摇杆前后：`x` 方向速度
   - 左摇杆左右：`y` 方向速度
   - 右摇杆左右：`yaw` 角速度
5. 任意时刻按 `select`
   - 进入阻尼并退出程序

---

## 一条命令总结

如果你的 H1 带步态 DWAQ checkpoint 已经训练好，推荐流程就是：

```bash
# 训练侧导出
cd /TienKung-Lab
python legged_lab/scripts/export_dwaq_policy.py \
    --checkpoint logs/h1_dwaq/<run_name>/model_<iter>.pt \
   --num_obs 40 \
   --num_actions 10 \
    --history_length 5

# 拷贝到部署工程
cd /LeggedLabDeploy
mkdir -p policy/h1_dwaq_phase
cp /TienKung-Lab/logs/h1_dwaq/<run_name>/exported/policy.pt \
   policy/h1_dwaq_phase/

# 上机部署
python deploy.py --config_path configs/h1_dwaq_phase.yaml --net eno1
```
