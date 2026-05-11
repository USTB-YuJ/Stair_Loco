# H1 Loco-with-Depth 部署指南

本目录用于把 `legged_gym/scripts/play.py` 在 `well_stair` 任务下导出的策略
`logs/h1_loco/well_stair/exported/policy_depth_1.pt` 部署到 **MuJoCo 仿真**
和 **Unitree H1 实机**。该策略由 `PolicyExporterDepth`
(`legged_gym/utils/helpers.py`) 导出，是单阶段的本体感知 + 历史 + 深度图端到端策略。

---

## 1. 文件清单

| 路径 | 作用 |
|---|---|
| `resources/robots/h1_description/h1_camera.xml`   | H1 模型，`depth_cam` 外参来自 URDF `d435_left_imager_joint` |
| `resources/robots/h1_description/scene_camera.xml` | 包含上面模型 + 地面 + 一段台阶障碍 |
| `deploy_camera/deploy_mujoco/deploy_mujoco_h1_camera.py` | MuJoCo 部署脚本 |
| `deploy_camera/deploy_mujoco/configs/h1_camera.yaml`     | MuJoCo 部署配置 |
| `deploy_camera/deploy_real/deploy_real_h1_camera.py`     | 实机部署脚本 (Unitree SDK + RealSense) |
| `deploy_camera/deploy_real/configs/h1_camera.yaml`       | 实机部署配置 |

---

## 2. 策略接口（再确认一次）

```text
policy = torch.jit.load("policy_depth_1.pt")

actions = policy(
    obs,      # Tensor[B=1, 39]
    history,  # Tensor[B=1, 10, 39]
    depth,    # Tensor[B=1, D, 64, 64]
)             # -> Tensor[B=1, 10]   (action mean, 不含噪声)
```

### 2.1 `obs`（39 维，每控制周期更新一次）

来源：`H1_Loco_Robot.compute_observations`。

| 索引 | 维度 | 物理含义 | 缩放 |
|---|---|---|---|
| `[0:3]`   | 3 | 速度指令 `[vx, vy, wz]` | × `cmd_scale = [2.0, 2.0, 0.25]` |
| `[3:6]`   | 3 | 机身坐标系下角速度 | × `ang_vel_scale = 0.25` |
| `[6:9]`   | 3 | 投影重力 (`projected_gravity`) | — |
| `[9:19]`  | 10 | `dof_pos − default_angles` | × `dof_pos_scale = 1.0` |
| `[19:29]` | 10 | `dof_vel` | × `dof_vel_scale = 0.05` |
| `[29:39]` | 10 | 上一步策略输出的 `action` | — |

> 关节顺序（policy 内部约定，与 IsaacGym 训练时 `default_joint_angles` 中前 10 项一致）：
> ```
> 0  left_hip_yaw     5  right_hip_yaw
> 1  left_hip_roll    6  right_hip_roll
> 2  left_hip_pitch   7  right_hip_pitch
> 3  left_knee        8  right_knee
> 4  left_ankle       9  right_ankle
> ```

### 2.2 `history`（10 × 39）

把过去 10 帧 `obs` 按时间顺序堆叠，最新帧放在 `history[:, -1]`。
脚本里维护一个张量 `trajectory_history`，每个控制步：

```python
trajectory_history = torch.cat(
    [trajectory_history[:, 1:], obs.unsqueeze(1)], dim=1
)
```

### 2.3 `depth`（D × 64 × 64）

- D = `depth_buffer_len` in the config.

- 训练时 `cfg.depth.buffer_len = 10`、`update_interval = 5`，所以以 50 Hz 控制
  对应 **10 Hz 的深度刷新**，约覆盖 1s 深度历史。
- 缓存按时间顺序排列（最新帧在末尾），策略输入为
  `depth_buffer`（完整堆叠）。
- 深度像素值映射到 **`[-0.5, 0.5]`**：

  ```text
  d_clip = clip(d_meters, near=0, far=2)
  d_norm = d_clip / 2.0 − 0.5      # near→-0.5, far→0.5
  ```

- 相机外参（`camera_parent_body = torso_link`，与 `h1_description/urdf/h1.urdf` 一致）：
  - `position = (0.10848, 0.01750, 0.69317)` m（torso 系）
  - `y_angle = [50.8, 50.8]`，roll/yaw 固定为 0（不做相机外参随机化）
  - 当前训练侧 Warp ray 使用 FOV 路径生成射线；外参保持与实机机械安装一致。
  - `fovy_range = [79.3, 79.3]` 控制训练侧 Warp full frame 视锥。
  - 训练侧 Warp 以 **96 × 128**（H×W）作为 full frame，按
    `crop_pixels = [32, 32, 32, 0]` 裁出中心靠下 **64 × 64** ROI。
  - 部署侧 RealSense / MuJoCo 先从 **480 × 640**（H×W）resize 到
    **96 × 128**，再使用同一 `crop_pixels` 裁出 **64 × 64** ROI。

### 2.4 输出 `action`

10 维，按上面的关节顺序。下游：

```text
target_dof_pos = default_angles + action × action_scale     # action_scale = 0.25
torque         = kp × (target − q) − kd × dq               # 每物理步执行
```

---

## 3. MuJoCo 部署

### 3.1 安装依赖

```bash
pip install mujoco torch pyyaml opencv-python numpy
```

`legged_gym` 包需要可被导入（脚本里通过 `from legged_gym import LEGGED_GYM_ROOT_DIR`
解析 policy / xml 路径）。本仓库根目录已是 `LEGGED_GYM_ROOT_DIR`。

### 3.2 运行

```bash
cd /root/gpufree-data/workspace/more
python deploy_camera/deploy_mujoco/deploy_mujoco_h1_camera.py h1_camera.yaml
```

可选参数：
- 不传 `config` 时默认使用 `h1_camera.yaml`。
- 修改 `cmd_init: [vx, vy, wz]` 即可改变默认行进速度。
- `show_depth: true` 会用 OpenCV 弹一个 `depth_cam` 小窗口实时显示深度。

### 3.3 控制流（每个物理步）

```text
mj_step ──┬─► PD: tau = kp(target_dof_pos − q) − kd · dq
          │
          └─► 每 control_decimation (=10) 步触发一次：
                ┌─ 拼接 obs (39)
                ├─ 滚动 trajectory_history (10×39)
                ├─ 每 cam_update_interval (=5) 个控制步：
                │     渲染 depth_cam → 480×640 → resize 96×128 → 裁 ROI 64×64 → 归一化 → 推入 depth_buffer
                ├─ depth_in = depth_buffer
                ├─ action = policy(obs, history, depth_in)
                └─ target_dof_pos = default + action × 0.25
```

### 3.4 时间频率对照

| 频率 | 周期 | 触发 |
|---|---|---|
| 物理步      | `simulation_dt = 0.002 s` | `mj_step` + PD |
| 策略推理    | `0.02 s` (50 Hz)           | 每 10 个物理步 |
| 深度刷新    | `0.1 s`  (10 Hz)           | 每 5 次推理 |

> 这与训练时一致：IsaacGym 中 `decimation = 4`、`sim dt = 0.005`、
> `update_interval = 5`，等效控制频率与深度刷新频率相同。

---

## 4. 实机部署（Unitree H1 + RealSense D435i）

### 4.1 硬件假设

- Unitree H1 + 厂家原版控制器；通过以太网与上位机直连。
- 头部/躯干按 URDF 安装一台 Intel RealSense（推荐 D435i），外参与训练
  `depth.position` / `y_angle` 一致，并使用同一 resize + ROI 流程（见上表与 `h1_camera.xml`）。
- 上位机能 import `unitree_sdk2py` 与 `pyrealsense2`，且具备 root 权限或
  能访问相应的 USB / 网络设备。

### 4.2 安装依赖

```bash
pip install torch numpy pyyaml opencv-python pyrealsense2 scipy
# unitree_sdk2py 按官方文档安装（与 deploy_camera/deploy_real/deploy_real.py 共用）
```

### 4.3 启动

```bash
cd /root/gpufree-data/workspace/more/deploy_camera/deploy_real
python deploy_real_h1_camera.py {net_interface} h1_camera.yaml
```

例如：

```bash
python deploy_real_h1_camera.py enp3s0 h1_camera.yaml
```

### 4.4 启动流程（与原 `deploy_real.py` 一致）

1. **零力矩状态**：脚本启动后机器人电机进入零力矩。挂吊后用手晃动确认。
2. **进入默认姿态**：按遥控器 `START`，机器人 2 秒内插值到 `default_angles`
   定义的腿部姿态（双臂/腰保持零位）。
3. **保持默认姿态**：按 `A` 进入策略闭环。建议确认机器人脚已稳定接触地面。
4. **策略闭环**：
   - 左摇杆前后 → `vx`
   - 左摇杆左右 → `vy`（注意符号已取反，与默认 H1 习惯一致）
   - 右摇杆左右 → `wz`
   - 摇杆原始范围 `[-1, 1]` × `max_cmd = [0.8, 0.5, 1.57]` → 物理速度
   - 物理速度再乘 `cmd_scale = [2, 2, 0.25]` 写入 `obs[0:3]`，与训练分布一致。
5. **退出**：按 `SELECT` 或 `Ctrl+C`，机器人切到 `damping` 模式后退出。

### 4.5 RealSense 深度处理

`RealSenseDepth` 在后台线程持续抓帧：

```text
raw uint16 (mm)  →  meters (×depth_scale)
                  →  resize 到 96×128 full frame
                  →  按 [32,32,32,0] 裁出 64×64 ROI，与训练 `depth.resized` 一致
                  →  无效值 (==0) 视为远端 (= far_clip)
                  →  clip 到 [0, 2] m
                  →  归一化到 [-0.5, 0.5]
```

主循环按 `cam_update_interval` 拉取最新一张归一化深度帧，推入 `depth_buffer`。

### 4.6 关节索引映射 `leg_joint2motor_idx`

YAML 中 `leg_joint2motor_idx` 把策略关节顺序 (0..9) 映射到 H1 SDK 电机编号。
默认值复用了 `deploy_camera/deploy_real/configs/h1.yaml` 中的映射：

```yaml
leg_joint2motor_idx: [7, 3, 4, 5, 10,  8, 0, 1, 2, 11]
```

含义：策略输出的第 0 维（`left_hip_yaw`）对应 SDK 第 7 号电机，等等。
**部署前请用具体的 H1 固件版本进行核对**——不同批次的 H1 SDK 编号不一定相同。

---

## 5. 与训练侧的一一对照

| 项目 | 训练 (`H1_Loco_Cfg`) | 部署 |
|---|---|---|
| `num_actions` | 10 | 10 |
| `num_observations` | 39 | 39 |
| `obs_history_len` | 10 | 10 |
| `commands_scale` | `[2, 2, 0.25]` | `cmd_scale: [2, 2, 0.25]` |
| `obs_scales.ang_vel` | 0.25 | `ang_vel_scale: 0.25` |
| `obs_scales.dof_pos` | 1.0 | `dof_pos_scale: 1.0` |
| `obs_scales.dof_vel` | 0.05 | `dof_vel_scale: 0.05` |
| `control.action_scale` | 0.25 | `action_scale: 0.25` |
| `control.stiffness` | hip_yaw/roll/pitch=150, knee=200, ankle=40 | `kps: [150, 150, 150, 200, 40, ...]` |
| `control.damping`   | hip_yaw/roll/pitch=2, knee=4, ankle=2 | `kds: [2, 2, 2, 4, 2, ...]` |
| `decimation` | 4 (sim_dt 5 ms) | mujoco: `simulation_dt=0.002, control_decimation=10` (=20 ms) |
| `depth.buffer_len` | 10 | `depth_buffer_len: 10` |
| `depth.update_interval` | 5 (= 100 ms @ 50 Hz) | `cam_update_interval: 5` |
| `depth.near_clip / far_clip` | 0 / 2 m | `depth_near_clip / depth_far_clip` |
| `depth.original / resized` | (96, 128) / (64, 64) | RealSense / MuJoCo: 480×640 → resize 96×128 → crop [32,32,32,0] → 64×64 |
| `depth.fovy_range` | [79.3, 79.3] | 训练 Warp FOV 路径；部署侧通过 480×640 → 96×128 → ROI 对齐 |
| `depth.position` | (0.10848, 0.01750, 0.69317) torso 系 | MuJoCo `<camera pos=...>`；实机机械对位 |
| `depth.y_angle` | [50.8, 50.8]° | MuJoCo: `euler="0 -0.886627 0"`；实机俯仰 |

---

## 6. 排查建议

- **机器人立刻摔倒 / 频繁出轨**
  - 确认 `leg_joint2motor_idx` 与你的 H1 固件匹配（特别是从 `default_pos_state`
    手按 `A` 后看双脚是否对应正确）。
  - 确认 IMU 已经从 torso 转到 pelvis 坐标（`imu_type: "torso"` + `transform_imu_data`）。
  - 把 `cmd_init` 调成 `[0, 0, 0]`、`max_cmd` 临时调小，看原地踏步是否稳定。
- **深度图异常**
  - 用 `cv2.imshow(..., depth_buffer[0, -1] + 0.5)` 在 mujoco 端先确认渲染是“前下方”；
    实机端可以临时把 `RealSenseDepth.read()` 的内容存图调试。
  - 若 RealSense 与训练 FOV 仍有偏差，优先机械对位；必要时再调整
    `depth_crop_pixels`；默认是 **480×640 → 96×128 → 中心靠下 64×64 (H×W) ROI**。
- **JIT 加载报错**
  - 该 PT 是 `torch.jit.script` 导出，需要 PyTorch 版本 ≥ 2.0；建议与训练侧
    `rsl_rl` 所依赖的版本保持一致。
---

## 7. 关于自遮挡的重要说明

训练时使用的 **Warp 深度渲染器只渲染静态地形 mesh，不渲染机器人自身**。
换句话说：

- 一阶段 (`well_stair`) 的 `policy_depth_1.pt` **从未在带自遮挡的深度图上训练过**。

但是 **MuJoCo `Renderer.enable_depth_rendering()` 默认会渲染所有 geom**，
包括机器人自身——头/胸高度俯视的相机会看到自己的大腿/小腿/(部分)手臂，
它们距离相机只有几十厘米，归一化后接近 ‑0.5。这就是你在仿真里看到的
"稳定 mask"，对一阶段策略来说属于 **out-of-distribution** 输入。

### 7.1 mujoco 端的处理（已在脚本里实现）

`scene_camera.xml` 把地形 (`floor`、`step*`) 放到 `group=2`，机器人 mesh
保持原先的 `group=0/1`。`deploy_mujoco_h1_camera.py` 在渲染深度时构造一个
`MjvOption`，只保留 `terrain_geom_group`：

```python
opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)
for g in range(len(opt.geomgroup)):
    opt.geomgroup[g] = 1 if g == terrain_group else 0
renderer.update_scene(data, camera=cam_id, scene_option=opt)
```

行为由 YAML 控制：

```yaml
hide_robot_in_depth: true     # 一阶段策略：必须 true
terrain_geom_group: 2
```

效果（实测，pelvis 处俯视 45°，H1 站立姿态）：

| 渲染模式 | depth.min | depth.mean | 远端占比 |
|---|---|---|---|
| 完整场景（带机器人） | **‑0.488** (≈24 mm，自己的腿) | 0.224 | — |
| 仅地形 (`hide_robot_in_depth: true`) | **0.076** (≈1.15 m) | 0.380 | **51%** |

后者才是策略训练时见过的分布。

> 训练侧若启用自遮挡渲染（per-env BVH），可把 `hide_robot_in_depth`
> 改成 `false`，让 mujoco 端把机器人也渲染进深度图，以贴近真机的 RealSense
> 输入。

### 7.2 深度图朝向（MuJoCo 必须 `rot90 k=1`）

MuJoCo 的相机坐标约定是 `+X=右, +Y=上, 沿 -Z 看`。我们 XML 里只对相机绕 Y
轴俯仰 -45° (`euler="0 -0.7854 0"`)，没有绕光轴的 roll，所以渲染出来的图：

| 图像方向 | = MuJoCo 相机轴 | = 机身坐标 |
|---|---|---|
| 右   | +X_cam | (forward + up) |
| 上   | +Y_cam | **机身左侧 (+Y body)** |

也就是说原始 mujoco 输出把"机身左侧"放到了图像顶部——视觉上像被旋转了 90°。
而训练用的 Warp 渲染器是自然取向（top = horizon ahead, left = robot's left）。
策略只见过自然取向的输入，所以部署时必须把 mujoco 渲染结果 **逆时针转 90°**：

```python
raw = renderer.render()
raw = np.rot90(raw, k=1)
```

这件事 `deploy_mujoco_h1_camera.py` 已通过 yaml 的 `depth_rot90_k: 1` 自动处理。
实测把 `k` 设为 0 / 1 / -1 / 2，将"在机身左侧的高块"分别画到 TL / **BL** / TR / BR——
唯独 `k=1` 时既在图像左半（左→左）又在下半（地面物→图像下方），与"前下视相机"
的自然取向一致。

### 7.3 真机相机的取向

真机端 `deploy_real_h1_camera.py` 的 `RealSenseDepth` 也开放了
`depth_rot90_k`（默认 0），原因是大多数 RealSense（D435i 等）正装
（USB 口朝下）时输出本身就符合自然取向。如果你把相机物理旋转了：

| 物理安装 | `depth_rot90_k` |
|---|---|
| 正装（USB 口朝下） | `0` |
| 逆时针转 90° | `1` |
| 顺时针转 90° | `-1` |
| 倒装 | `2` |

### 7.4 真机端的不可避免遮挡

真机 RealSense 一定会拍到自己的腿/躯干，没法通过软件"隐藏"。因此对当前
**未在自遮挡深度图上训练**的策略，真机部署有以下几种缓解办法（按优先级）：

1. **优先在训练侧引入自遮挡 / 遮挡增强**——例如启用 per-env BVH 自遮挡渲染，
  或对遮挡分布做数据增强后重新训练。
2. **机械方面把相机抬高/前移/俯角加大**，让机器人自己的腿尽量挪出 FOV。
3. **软件遮挡增强**：在 `RealSenseDepth._loop` 里对固定区域做占位/平滑处理，
  仅用于调试定位遮挡问题，**不推荐作为长期方案**。

简单地说：**当前你看到的稳定 mask 确实会影响部署**。mujoco 端我们通过隐藏
机器人 geom 把训练-测试分布对齐了；真机端更稳妥的做法是让训练分布覆盖自遮挡
（例如启用 per-env BVH 或加入遮挡增强）。
