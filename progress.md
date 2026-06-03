# H1 DWAQ Progress

更新时间：2026-05-30 23:07 CST
工作区：`/root/gpufree-data/workspace/G1DWAQ_Lab-main`
当前分支：`dwaq`，相对 `origin/dwaq` 超前 4 个提交；另有 DWAQ aux heads 相关未提交改动。

## 当前结论

这个仓库当前主线是面向 Unitree H1 的 DWAQ 盲行走/带步态相位控制框架。训练侧、Isaac play、MuJoCo sim2sim、TorchScript 导出和部署侧配置都已经建立起来；重点已经从 H1 适配、10 自由度下肢控制和旋转指令优化，推进到“隐式台阶理解 + 受绊后主动抬腿”的稳定性增强。

最新结论：`post_stumble_lift` 奖励结合 DWAQ latent 辅助监督后，最新 H1 实验反馈效果不错；当前 sim2sim 脚本已经指向 `logs/h1_dwaq/2026-05-30_12-00-27/model_7000.pt` 作为新的重点验证模型。部署侧 `num_obs=43` 和 H1 DWAQ policy 文件已同步，推理接口保持 `obs=43, hist=5, action=10`。

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
- 2026-05-29 `74b5f43` 增加 `post_stumble_lift` 受绊后抬腿奖励。
- 2026-05-30 工作区未提交：实现 DWAQ 隐式台阶/抬腿辅助监督 v1，包括 env 标签、latent aux heads、storage/PPO/runner 数据流和日志指标。

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
- `post_stumble_lift`：轻微受绊后，在短时间窗口内奖励脚高度上升，当前 H1 权重为 `0.5`。

### DWAQ 隐式台阶/抬腿辅助监督

2026-05-30 已实现训练期 auxiliary supervision，目标是让 `latent(16)` 更明确地携带“刚被绊、需要抬多高、当前地形状态”等信息。部署接口不变，actor 推理仍然只走 `obs_history -> encoder -> actor`。

- Env 侧新增 `dwaq_aux_targets`，格式固定为 5 维：`[stumble_l, stumble_r, clearance_l, clearance_r, terrain_state]`。
- `stumble_recent`：检测摆动腿水平接触力主导的轻微绊脚，并保持约 `0.25s`。
- `required_clearance`：复用 `forward_obstacle_height`，目标为 `0.08 + obstacle + obstacle_bonus`，上限 `0.30m`。
- `terrain_state`：`0=flat/non-stair, 1=stairs_up, 2=stairs_down, 3=other_rough`；优先用 command generator 的 terrain masks，G1 默认 command 路径也有 scene terrain fallback。
- `ActorCritic_DWAQ` 从 `mean_latent` 接三组训练期 head：`stumble_head`、`clearance_head`、`terrain_head`。
- `DWAQPPO` 新增 BCE / SmoothL1 / CrossEntropy 辅助 loss，并用 `aux_ramp_iterations=1000` 从 0 线性升权。
- 旧 checkpoint 允许缺少 aux head 权重；缺失时打印 warning，辅助 head 随机初始化。

需要重点观察的新增 TensorBoard 指标：

- `Loss/aux_stumble`、`Aux/stumble_positive_rate`
- `Loss/aux_clearance`、`Aux/clearance_mae`
- `Loss/aux_terrain`、`Aux/terrain_acc`
- `Loss/aux_total`、`Loss/aux_ramp`

## 产物与验证状态

已有 H1 DWAQ 训练日志和 checkpoint，主要目录包括：

- `TienKung-Lab/logs/h1_dwaq/2026-04-17_10-22-42/`
- `TienKung-Lab/logs/h1_dwaq/2026-04-25_00-09-54/`
- `TienKung-Lab/logs/h1_dwaq/2026-04-25_11-14-02/`
- `TienKung-Lab/logs/h1_dwaq/2026-05-28_09-51-22/`
- `TienKung-Lab/logs/h1_dwaq/2026-05-29_16-34-09/`：包含 `post_stumble_lift` 奖励后的 H1 训练，`model_7700.pt` 通过旧 checkpoint 兼容检查。
- `TienKung-Lab/logs/h1_dwaq/2026-05-30_12-00-27/`：最新效果较好的实验，当前 sim2sim 指向 `model_7000.pt`。

已导出的 TorchScript policy：

- `TienKung-Lab/logs/h1_dwaq/2026-04-17_10-22-42/exported/policy.pt`
- `TienKung-Lab/logs/h1_dwaq/2026-04-25_00-09-54/exported/policy.pt`

最新重点实验：

- run：`TienKung-Lab/logs/h1_dwaq/2026-05-30_12-00-27/`
- 当前验证模型：`model_7000.pt`，已写入 `sim2sim.sh`。
- 定性反馈：最新实验效果不错，说明受绊抬腿奖励和隐式台阶/抬腿辅助监督这条路线值得继续推进。
- 备注：`2026-05-30_11-34-49/` 是小规模 smoke run，只用于检查训练数据流，不作为效果对比主 run。

建议重点观察 TensorBoard：

- `Command/error_vel_yaw`
- `Command/yaw_only_mode_rate`
- `Command/arc_turn_mode_rate`
- `Curriculum/target_success_rate`
- `Loss/aux_stumble`、`Aux/stumble_positive_rate`
- `Loss/aux_clearance`、`Aux/clearance_mae`
- `Loss/aux_terrain`、`Aux/terrain_acc`
- 摔倒率、termination、feet slide、feet stumble、post-stumble lift 相关指标

## 当前脚本指向

- `train.sh`：训练 `h1_dwaq`，4096 env，50000 iterations。
- `play.sh`：加载 `2026-05-29_16-34-09/model_7700.pt`，`difficulty=0.1`。
- `sim2sim.sh`：加载 `2026-05-30_12-00-27/model_7000.pt`，使用 H1 MuJoCo scene。
- `export_dwaq_policy.sh`：仍加载 `2026-05-28_09-51-22/model_9000.pt`，按 `num_obs=43, num_actions=10, history_length=5` 导出；如果确认 2026-05-30 模型更优，后续应同步更新导出 checkpoint。

## 待处理事项

1. 继续围绕 `2026-05-30_12-00-27/model_7000.pt` 做系统验证。
   - MuJoCo sim2sim 重点看上台阶是否更少卡脚、轻微绊脚后是否会快速抬腿恢复。
   - Isaac play 继续覆盖原地转、弧线转、大转角目标点和不同台阶难度。

2. 若确认 2026-05-30 模型稳定优于旧模型，更新导出链路。
   - 将 `export_dwaq_policy.sh` 的 checkpoint 切到新的优选模型。
   - 重新导出并同步到 `LeggedLabDeploy/policy/h1_dwaq_phase/policy.pt`。

3. 跟踪 aux head 是否真的被 latent 学到。
   - `Aux/clearance_mae` 应逐步下降，单位近似为米。
   - `Aux/terrain_acc` 应明显高于随机四分类。
   - `Aux/stumble_positive_rate` 不能长期为 0，否则说明受绊正样本覆盖不足。

4. 建议整理 shell 脚本格式。
   - 多个脚本末尾仍有多余反斜杠或缺少换行。
   - 不一定影响当前运行，但会降低后续追加参数时的可维护性。

5. 旋转优化暂时保留当前参数；如果后续模型仍弱，再小步调参。
   - 可将 `yaw_only_probability` 从 `0.2` 提到 `0.25`。
   - 可将 `arc_turn_probability` 从 `0.3` 调到 `0.35`。
   - yaw 范围可从 `(-0.8, 0.8)` 逐步扩到 `(-1.0, 1.0)`。
