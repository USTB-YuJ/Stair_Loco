# H1 DWAQ Progress

更新时间：2026-05-28 10:00 CST  
工作区：`/root/gpufree-data/workspace/G1DWAQ_Lab-main`  
当前分支：`dwaq`，相对 `origin/dwaq` 超前 3 个提交。

## 当前结论

这个仓库当前主线是面向 Unitree H1 的 DWAQ 盲行走/带步态相位控制框架。训练侧、Isaac play、MuJoCo sim2sim、TorchScript 导出和部署侧配置都已经建立起来；最近的重点从 H1 适配推进到 10 自由度下肢控制、目标点速度命令，以及旋转指令优化。

当前仍处在训练验证阶段：2026-05-28 已启动一轮包含旋转优化的新训练，部署侧还需要同步 `num_obs=43`、复制导出的 H1 DWAQ policy，并完成 sim2sim/实机前验证。

## Git 时间线

> 来源：`git log --reverse --date=short --pretty=format:"%ad %h %s"`

- 2026-04-14 `a545a95` 增加 H1-2 URDF 以及参数奖励调整。
- 2026-04-14 `69c9e43` 调整 ankle PD，尝试处理训练中的垫脚问题。
- 2026-04-14 `6404627` 增加显卡序号指定。
- 2026-04-14 `dd28805` 根据质量映射修正奖励权重。
- 2026-04-15 `a445bee` 增加 H1 资产与环境，调整奖励函数，优化 reset 条件。
- 2026-04-15 `ea24025` 增加手臂静止惩罚。
- 2026-04-16 `7fe6101` 更新 bash 文件以适配 H1。
- 2026-04-16 `72135f1` 合并 `origin/dwaq` 到 `dwaq`。
- 2026-04-16 `312507c` 腰部和手臂静止惩罚改为全时段生效。
- 2026-04-16 `56acb51` 增加正向直立奖励。
- 2026-04-16 `f2a7a02` 调整接触力 reset 条件，并把 H1 台阶加入 MuJoCo。
- 2026-04-17 `0226130` 调整速度跟踪奖励以及随机化方案。
- 2026-04-17 `8ad6f72` 完成部署部分的 H1 适配化，包括 `h1_dwaq_phase.yaml`、部署文档和导出/play/sim2sim 脚本整理。
- 2026-04-24 `7716f97` 修改为 10 自由度控制，策略只控制 H1 下肢，腰部和上肢保持默认位姿。
- 2026-04-24 `78651c4` 将目标点任务改造成速度命令，加入 target-point command、目标进度/成功率统计和 terrain curriculum 逻辑。
- 2026-05-28 `f30e448` 优化旋转问题奖励：补 yaw-only/arc-turn 命令分布，增强 yaw tracking，转向时放松 gait/hip 约束。

## 当前框架状态

### 训练侧

- 主任务：`h1_dwaq`
- 训练入口：`TienKung-Lab/legged_lab/scripts/train.py`
- 快捷脚本：`TienKung-Lab/train.sh`
- 当前训练命令：

```bash
cd /root/gpufree-data/workspace/G1DWAQ_Lab-main/TienKung-Lab
bash train.sh
```

`train.sh` 当前等价于：

```bash
python legged_lab/scripts/train.py \
    --task=h1_dwaq \
    --gpu="0" \
    --logger=tensorboard \
    --num_envs=4096 \
    --max_iterations=50000 \
    --headless
```

当前 H1 DWAQ 的核心设定：

- 动作维度：10，只控制下肢。
- 固定关节：腰部、肩、肘等上肢关节保持默认位姿。
- DWAQ 历史长度：5。
- 当前单帧 actor observation：43 维。
- 观测组成：`ang_vel(3) + projected_gravity(3) + command(3) + joint_pos(10) + joint_vel(10) + last_action(10) + gait_phase(4)`。

### 命令分布

当前 `TargetPointVelocityCommand` 支持三类非站立命令：

- `target`：常规目标点导航命令。
- `yaw-only`：原地旋转，默认占非站立样本的 20%。
- `arc-turn`：带前进速度的弧线转向，默认占非站立样本的 30%。

配置入口在 `TargetPointCommandCfg`：

```python
yaw_only_probability = 0.2
arc_turn_probability = 0.3
yaw_only_ang_vel_z = (-0.8, 0.8)
yaw_only_min_abs_ang_vel_z = 0.25
arc_turn_lin_vel_x = (0.1, 0.5)
arc_turn_ang_vel_z = (-0.8, 0.8)
arc_turn_min_abs_ang_vel_z = 0.2
```

目标点 curriculum 只统计 `target` 模式，`yaw-only` 和 `arc-turn` 不会污染目标点成功率/超时率。

### 奖励侧

当前 H1 DWAQ 在基础 H1 reward 上叠加了这些重点：

- `track_ang_vel_z_exp`：yaw 跟踪增强为 `weight=3.0, std=0.4`。
- `joint_deviation_hip`：使用 `joint_deviation_l1_turn_relaxed`，转向时降低 hip yaw/roll 对默认位姿的束缚。
- `gait_phase_contact`：使用 `terrain_scaled_gait_phase_contact`，楼梯和高 yaw 指令下都会降低 gait phase 接触引导强度。
- `target_goal_progress` / `target_goal_reached`：鼓励目标点进度和到达。
- `terrain_aware_feet_swing_height`：根据脚下地形和前方障碍调整摆腿高度惩罚。

## 产物与验证状态

已有 H1 DWAQ 训练日志和 checkpoint，主要目录包括：

- `TienKung-Lab/logs/h1_dwaq/2026-04-17_10-22-42/`
- `TienKung-Lab/logs/h1_dwaq/2026-04-25_00-09-54/`
- `TienKung-Lab/logs/h1_dwaq/2026-04-25_11-14-02/`
- `TienKung-Lab/logs/h1_dwaq/2026-05-28_09-51-22/`

已导出的 TorchScript policy：

- `TienKung-Lab/logs/h1_dwaq/2026-04-17_10-22-42/exported/policy.pt`
- `TienKung-Lab/logs/h1_dwaq/2026-04-25_00-09-54/exported/policy.pt`

当前正在运行的新训练：

- run：`TienKung-Lab/logs/h1_dwaq/2026-05-28_09-51-22/`
- 进程：`python legged_lab/scripts/train.py --task=h1_dwaq --gpu=0 --logger=tensorboard --num_envs=4096 --max_iterations=50000 --headless`
- 截至 2026-05-28 10:00 CST，已看到 `model_0.pt`、`model_100.pt`、`model_200.pt` 和 TensorBoard event 文件。

建议重点观察 TensorBoard：

- `Command/error_vel_yaw`
- `Command/yaw_only_mode_rate`
- `Command/arc_turn_mode_rate`
- `Curriculum/target_success_rate`
- 摔倒率、termination、feet slide、feet stumble 相关指标

## 当前脚本指向

- `train.sh`：训练 `h1_dwaq`，4096 env，50000 iterations。
- `play.sh`：加载 `2026-04-25_00-09-54/model_14100.pt`。
- `sim2sim.sh`：加载 `2026-04-25_11-14-02/model_7600.pt`。
- `export_dwaq_policy.sh`：加载 `2026-04-25_00-09-54/model_15000.pt`，并按 `num_obs=43, num_actions=10, history_length=5` 导出。

## 待处理事项

1. 部署配置仍需同步观测维度。
   - `LeggedLabDeploy/configs/h1_dwaq_phase.yaml` 当前仍是 `num_obs: 40`。
   - 当前训练/导出链路已经是 `num_obs=43`。
   - 部署前必须改为 `num_obs: 43`，否则 TorchScript 输入维度会不匹配。

2. 部署目录缺少 H1 DWAQ policy。
   - 目标路径应为 `LeggedLabDeploy/policy/h1_dwaq_phase/policy.pt`。
   - 当前该文件尚未落在部署目录。

3. 需要用 2026-05-28 的旋转优化训练结果重新 play/sim2sim。
   - 先在 Isaac play 中测原地转、弧线转、大转角目标点。
   - 再用 MuJoCo sim2sim 检查 yaw 跟踪、足端滑移和姿态稳定性。

4. 建议整理 shell 脚本格式。
   - 多个脚本末尾仍有多余反斜杠或缺少换行。
   - 不一定影响当前运行，但会降低后续追加参数时的可维护性。

5. 若旋转仍弱，可继续调参。
   - 将 `yaw_only_probability` 从 `0.2` 提到 `0.25`。
   - 将 `arc_turn_probability` 从 `0.3` 调到 `0.35`。
   - 将 yaw 范围从 `(-0.8, 0.8)` 逐步扩到 `(-1.0, 1.0)`，不要一次跳太大。
   - 如果转向时步态被锁住，可继续降低 `turn_scale`，例如 `0.5 -> 0.35`。
