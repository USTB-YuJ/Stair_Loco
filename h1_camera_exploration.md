# H1 带深度相机策略：探索与优化记录

本文档记录了在 Unitree H1 人形机器人上应用带深度相机的 locomotion 策略过程中，
我们所做的分析、发现的问题、以及实施的改动。按时间和逻辑顺序编排，方便后续维护
和回顾。

---

## 目录

1. [策略分析：导出的 PT 权重接口](#1-策略分析导出的-pt-权重接口)
2. [部署代码搭建](#2-部署代码搭建)
3. [Body Mask 与自遮挡问题](#3-body-mask-与自遮挡问题)
4. [深度图朝向修复（rot90）](#4-深度图朝向修复rot90)
5. [框架问题澄清：AMP / 相机噪声 / 训练时渲染本体](#5-框架问题澄清)
6. [Per-Env 全身 BVH：设计与实现](#6-per-env-全身-bvh设计与实现)
7. [相机内外参与频率对齐分析](#7-相机内外参与频率对齐分析)
8. [训练启动参数建议](#8-训练启动参数建议)
9. [文件改动汇总](#9-文件改动汇总)
10. [待办 / 后续方向](#10-待办--后续方向)

---

## 1. 策略分析：导出的 PT 权重接口

### 背景

`play.py` 通过 `export_policy_as_jit_depth()` 将训练好的策略（`well_stair`
任务）导出为 `policy_depth_1.pt`。该策略是**一阶段**训练产物（没有做二阶段
residual / MoE），使用 `PolicyExporterDepth` 封装。

### 结论

导出的 JIT 模块包含三个子网络：`actor` + `history_encoder` + `depth_encoder`。

**签名**：
```python
actions = policy(obs, history, depth)
# obs:     Tensor[1, 39]       - 当前帧本体感知
# history: Tensor[1, 10, 39]   - 过去 10 帧本体感知
# depth:   Tensor[1, 2, 48, 64]- 2 帧深度图（3 帧 buffer 的前 2 帧）
# actions: Tensor[1, 10]       - 10 个腿关节动作增量
```

**obs 39 维排布**：
| 区间 | 维度 | 含义 |
|---|---|---|
| `[0:3]` | 3 | `commands * [2.0, 2.0, 0.25]` (vx, vy, wz) |
| `[3:6]` | 3 | `base_ang_vel * 0.25` |
| `[6:9]` | 3 | `projected_gravity` |
| `[9:19]` | 10 | `(dof_pos - default) * 1.0` |
| `[19:29]` | 10 | `dof_vel * 0.05` |
| `[29:39]` | 10 | `last_action` |

关节顺序（10 DOF，双腿）：
```
0 left_hip_yaw     5 right_hip_yaw
1 left_hip_roll    6 right_hip_roll
2 left_hip_pitch   7 right_hip_pitch
3 left_knee        8 right_knee
4 left_ankle       9 right_ankle
```

**输出后处理**：`target_dof_pos = default_angles + action * 0.25`，再走 PD 控制。

---

## 2. 部署代码搭建

### 新增文件

| 文件 | 用途 |
|---|---|
| `resources/robots/h1/h1_camera.xml` | H1 MuJoCo 模型 + pelvis 深度相机 |
| `resources/robots/h1/scene_camera.xml` | 含台阶障碍的场景（地形 geom 放 group=2） |
| `deploy_camera/deploy_mujoco/deploy_mujoco_h1_camera.py` | MuJoCo 部署脚本 |
| `deploy_camera/deploy_mujoco/configs/h1_camera.yaml` | MuJoCo 部署配置 |
| `deploy_camera/deploy_real/deploy_real_h1_camera.py` | 实机部署（Unitree SDK + RealSense） |
| `deploy_camera/deploy_real/configs/h1_camera.yaml` | 实机部署配置 |
| `deploy_camera/README_h1_camera.md` | 中文使用文档 |

### 关键设计

- **深度归一化**与训练侧一致：`d_norm = clip(d, 0, 2) / 2 - 0.5`，范围 `[-0.5, 0.5]`。
- **深度 buffer** 长度 3，策略只用 `[:, :2]`（最老 2 帧），模拟 100-200 ms 延迟。
- **`trajectory_history`**：长度 10 的滚动窗口，每控制步更新。
- **MuJoCo 渲染**需要 `rot90(k=1)` 修正朝向（见下文第 4 节）。
- 实机使用 `RealSenseDepth` 后台线程抓帧 + 归一化，支持 `depth_rot90_k` 配置。

---

## 3. Body Mask 与自遮挡问题

### 发现

在 MuJoCo 仿真中观察到深度图底部出现**稳定的黑色区域**，这是 MuJoCo
渲染器把 H1 自身的腿/躯干也渲染进了画面（距离相机几十厘米，归一化后接近 -0.5）。

### 根因分析

| 渲染路径 | 是否包含机器人本体 | 训练时用的 |
|---|---|---|
| Warp 渲染器 (`warp_camera=True`) | 只渲染地形 mesh | 是 |
| IsaacGym 原生 (`use_camera=True`) | 渲染全场景含机器人 | 否 |
| MuJoCo (`Renderer`) | 渲染全场景含机器人 | — |

训练时的 Warp 渲染器调用 `render_mesh(terrain.vertices, terrain.triangles)`，
**只塞了地形 mesh**，所以 `policy_depth_1.pt` 的 `depth_encoder` **从未见过自遮挡**。

仓库中的 `add_body_mask` 机制是为 g1 二阶段 residual 训练准备的，H1
当前配置为 `False`，body_mask 数据集也没有 H1 的。

### 临时修复（MuJoCo 部署侧）

通过 `MjvOption.geomgroup` 在深度渲染时隐藏机器人 geom（只保留 terrain group=2），
使 MuJoCo 输出与训练时的 Warp 输出分布一致。

配置：`hide_robot_in_depth: true`，`terrain_geom_group: 2`。

### 根本方案

→ 见第 6 节：Per-Env 全身 BVH。

---

## 4. 深度图朝向修复（rot90）

### 发现

MuJoCo 渲染的深度图被**旋转了 90 度**。

### 根因

MuJoCo 相机约定 `+X=右, +Y=上, 沿 -Z 看`。XML 中只做了绕 Y 轴的俯仰
`euler="0 -0.7854 0"`，没有绕光轴的 roll。结果 `+Y_cam`（图像"上"方向）
落在了机身 +Y（机身左侧），而非自然取向的"前方/上方"。

### 验证

用非对称障碍物（机身正前方 + 机身左侧各一个 box），对比 `rot90(k=0/1/-1/2)`：

| k | 左侧物体位置 | 判定 |
|---|---|---|
| 0 | TL (旋转了) | 错 |
| **1** | **BL (自然取向)** | **正确** |
| -1 | TR | 错 |
| 2 | BR | 错 |

### 修复

- `deploy_mujoco_h1_camera.py`：`render_depth(..., rot90_k=1)` 默认旋转。
- 配置 `depth_rot90_k: 1`（MuJoCo），`depth_rot90_k: 0`（RealSense 正装）。
- 参考：原仓库 `deploy_mujoco_with_resi.py:106` 也使用 `np.rot90(k=1)`。

---

## 5. 框架问题澄清

### 5.1 AMP 是否启用

**结论：H1 没有启用 AMP**。

`H1_Loco_CfgPPO.algorithm.use_amp = False`。Runner 类名虽然叫
`AMPOnPolicyRunnerMulti`，但内部所有 AMP 逻辑都被 `if self.use_amp` 守护。
H1 当前是纯 PPO + history encoder + depth encoder 的端到端策略。

### 5.2 相机噪声

**结论：H1 当前几乎全关**。

| 噪声类型 | H1 配置 | 效果 |
|---|---|---|
| `dis_noise` | `0` | 无全局距离偏移 |
| `gaussian_noise` | `False` | 无像素白噪声 |
| `gaussian_filter` | `False` | 无模糊 |
| `rand_position` | `False` | 无位置抖动 |
| `crop_depth` | `False` | 无裁剪 |
| `random_cam_delay` | `False` | 无延迟抖动 |
| `y_angle = [42, 48]` | ✅ 开 | 俯仰 ±3° 域随机化 |

**建议**：训练时开启 `dis_noise=0.03`、`gaussian_noise=True(std=0.02)`、
`gaussian_filter=True`、`rand_position=True`。

### 5.3 让相机看见本体

框架有两种方案看到本体：
1. `use_camera=True`（IsaacGym 原生 OpenGL）：自动渲染全场景含机器人，但
   并行度从 6000 降到 ~256-512。
2. **在 Warp 渲染器的射线碰撞目标里加入机器人 mesh**：保留 Warp 高并行度。

→ 最终采用方案 2，见第 6 节。

---

## 6. Per-Env 全身 BVH：设计与实现

### 6.1 设计思路

每个 env 拥有独立的 `wp.Mesh` BVH（仅包含该 env 自己的机器人简化几何）。
每条射线同时查询：
- **共享地形 BVH**（全局 1 个，静态）
- **per-env 机器人 BVH**（per-env 独立）

取两次 hit 的最近距离，自然产生自遮挡。跨 env 物理隔离。

```
ray → terrain BVH (共享) → t_terrain
ray → robot BVH[env_i]   → t_robot
depth = min(t_terrain, t_robot) * d_fwd
```

### 6.2 为什么要全身

即使 H1 的上半身（torso + arms）在默认站姿下不在 FOV 内，但：
- 机器人姿态变化（倾倒/剧烈运动）时手臂可能进入视野
- 真机相机外参漂移可能让本来 FOV 外的部位入镜
- 任何未建模的部位出现在画面中 = **OOD 输入**

### 6.3 简化几何模板（H1）

每个 link 用 box 或 capsule 近似：

| 部位 | link name | primitive | 三角面 |
|---|---|---|---|
| pelvis | pelvis | box | 12 |
| 胸 | torso_link | box | 12 |
| 大腿 ×2 | left/right_hip_pitch_link | capsule | 96 |
| 小腿 ×2 | left/right_knee_link | capsule | 96 |
| 脚 ×2 | left/right_ankle_link | box | 24 |
| 上臂 ×2 | left/right_shoulder_pitch_link | capsule | 96 |
| 前臂 ×2 | left/right_elbow_link | capsule | 96 |
| **合计** | **12 links** | | **432 tri/env** |

### 6.4 新文件

| 文件 | 作用 |
|---|---|
| `legged_gym/utils/robot_geom.py` | `LinkGeom` 数据类 + `build_box_mesh` / `build_capsule_mesh` / `build_robot_template` |
| `legged_gym/utils/h1_geom.py` | H1 全身 12-link 几何表 (`H1_LINK_GEOMS`) |
| `legged_gym/scripts/verify_self_occlusion_bvh.py` | 独立验证脚本（无需 IsaacGym） |

### 6.5 warp_render_v3.py 改动

- `DepthRendererWarp.__init__` 新增 `num_envs` / `far_t` / `miss_t`。
- 新增 `init_robot_meshes()`：预分配 `(num_envs * V, 3)` 大 verts buffer，
  创建 `num_envs` 个 `wp.Mesh`（sub-view 共享内存），收集 `mesh.id` 到
  `wp.array(dtype=wp.uint64)`。
- 新增 `update_robot_verts` kernel（一次 launch 变换所有 env 所有 link 的顶点）。
- 新增 `update_robot_meshes(rigid_body_states)`：launch kernel + loop refit。
  支持 `refit_stride` 跳帧。
- 新增 `draw_depth_dual` kernel：双 `mesh_query_ray`，`min(t_terrain, t_robot)`。
- 新增 `draw_depth_single` kernel：terrain-only fallback。
- `render_depth()` 自动选择 dual / single 路径。
- `mesh_query_ray` 的 max_t 从硬编码 `5.0` 改为 `far_t`（= far_clip + 0.2），
  作为跨 env 安全兜底。
- 移除 `pytorch3d` 依赖（替换为内联的 `_quaternion_wxyz_to_matrix`）。

### 6.6 legged_robot.py 改动

- `create_sim()` 末尾：`DepthRendererWarp` 构造时传入 `num_envs` / `far_t`；
  调用 `_init_self_occlusion_meshes()` 初始化 per-env BVH。
- `_create_envs()`：缓存 `self.body_names` 用于 link_name → body_index 映射。
- `_init_self_occlusion_meshes()`：新方法，`importlib.import_module(cfg.depth.robot_geom_module)`，
  `build_robot_template()`，解析 body indices，调 `init_robot_meshes()`。
- `warp_update_depth_buffer()`：在 `render_depth()` 前调
  `self.warp_renderer.update_robot_meshes(self.rigid_body_states)`。
- **删除** `add_body_mask` 分支（~12 行）。

### 6.7 配置改动

```python
# legged_robot_config.py (base, 默认 False)
class depth:
    enable_self_occlusion = False
    robot_geom_module = ""
    refit_stride = 1

# h1_loco_config.py (启用)
class depth(LeggedRobotCfg.depth):
    enable_self_occlusion = True
    robot_geom_module = "legged_gym.utils.h1_geom"
    refit_stride = 1
```

### 6.8 Body Mask 路径清理

- `legged_robot.py`：`warp_update_depth_buffer` 中 `add_body_mask` 整段删除。
- `g1_16dof_moe_residual_env.py`：body_masks `np.load` 删除。
- `moe_residual_on_policy_runner_multi.py`：iter > 30000 的 `add_body_mask = True` 删除。

### 6.9 跨 env 隔离验证

```
env 1 depth max-abs change after moving env 0 body in front of env 1's camera:
0.000000 m → PASS
```

### 6.10 性能基准（RTX 4090，432 tri/env 全身模板）

| envs | refit_stride=1 | stride=2 | stride=4 | stride=8 |
|---|---|---|---|---|
| 1024 | 443 ms | 245 ms | 146 ms | **88 ms** |
| 2048 | 885 ms | 570 ms | 265 ms | **175 ms** |
| 4096 | 1790 ms | 1030 ms | 576 ms | 360 ms |

**建议**：2048 envs + `refit_stride=8`（175 ms/tick），或同时把
`update_interval` 从 5 调大到 10（5 Hz depth，更贴近真机 RealSense 帧率）。

---

## 7. 相机内外参与频率对齐分析

### 7.1 内参

- **显式内参矩阵**（2026-04-27 更新后当前实现）：H1 Warp renderer 不再由 `fovy_range` 推导 ray，而是使用 RealSense 640x480 pinhole 内参 `fx=fy=384.77294921875`、`cx=324.17236328125`、`cy=236.48226928710938`。
- **等效视场**：这组内参约等价于 HFOV 79.5°、VFOV 63.9°；`fovy_range=[58,58]` 仅保留作兼容/文档字段，不再驱动 H1 Warp ray。
- **分辨率**：采集/渲染保持 RealSense 全帧 `(480, 640)`（H×W），策略输入为底部居中直接裁剪出的 `(48, 64)`（H×W）；部署侧不再使用 `cv2.resize` 下采样。
- 渲染输出是**平面深度** (Z-depth)，不是 range distance。

#### 与真机对照（当前实现）

| 参数 | 训练 | RealSense D435i |
|---|---|---|
| 内参 | `fx=fy=384.77294921875`, `cx=324.17236328125`, `cy=236.48226928710938` | 640x480 depth profile 应校验到同一组或等价值 |
| 等效 FOV | HFOV ≈79.5°，VFOV ≈63.9° | 由实机 profile 内参决定 |
| 纵横比 | 全帧 4:3，策略输入 48×64 crop | 4:3（640×480） |
| near/far | 0 / 2 m | 部署侧 clip 到 0 / 2 m |

**Checkpoint**：CNN 末端 **3×5** → `nn.Linear(64×3×5, 128)`；与旧 64×64 权重不兼容，需重新训练。

### 7.2 外参

- 位置 `(0.10848, 0.01750, 0.69317)` m（**`torso_link`** 系，URDF `d435_left_imager_joint`）。
- 俯仰固定 `y_angle=[50.8, 50.8]`°，Roll/Yaw 固定 `[0, 0]`。
- 当前 H1 配置关闭相机外参随机化，训练中所有 env 使用与真机安装一致的固定外参。

### 7.3 频率对齐

| 时钟 | 周期 | 频率 |
|---|---|---|
| 物理 step | 5 ms | 200 Hz |
| 控制 / 策略 step | 5 ms × 4(decimation) = 20 ms | 50 Hz |
| 深度重渲染 | 20 ms × 5(update_interval) = 100 ms | 10 Hz |

#### 对齐机制

1. `warp_update_depth_buffer()` 每 5 个控制步才真正渲染，新帧 append 到
   `warp_depth_buffer`（shape `[B, 3, 48, 64]`）的末尾，最老帧被挤掉。
2. `env.step()` 返回的 `extras["depth"]` 在**深度刷新步**是 buffer 的引用，
   其余步是 `None`。但 runner 持有上一次的引用，所以**每个控制步都用一份有效 depth**。
3. 策略 `actor_critic.act(obs, history, depth[:, :2, ...])` 只用 buffer
   的**前 2 帧**（最老 2 帧），故意丢弃最新帧。
4. 等效地：策略看到的深度**永远比当前状态延迟 100-200 ms**。
   这是人为设计，模拟真机 RealSense 的 pipeline 延迟。

#### 时间线

```
control step:  0    1    2    3    4    5    6    7    8    9   10
               ─────────────────────────────────────────────────►
50 Hz policy:  P    P    P    P    P    P    P    P    P    P    P
10 Hz depth:   D                        D                        D
               │                        │
           buffer=[f0,f1,f2]      buffer=[f1,f2,f3]
           policy sees [:,:2]     policy sees [:,:2]
             = [f0, f1]            = [f1, f2]
              (200ms, 100ms ago)    (200ms, 100ms ago)
```

#### 注意事项

- `global_counter` 从不 reset，深度刷新 phase 在所有 env 间**完全同步**。
- `random_cam_delay` 字段虽然存在，但**代码从未实现消费它**。
- Depth 不参与 critic（`evaluate(critic_obs, history)` 没有 depth 输入）。

---

## 8. 训练启动参数建议

### 必须改

```python
# train.py
args.resume = False         # 必须从头训（depth 分布变了）
args.load_run = ""          # 不 resume 旧 checkpoint
```

### 性能调整

```python
# h1_loco_config.py -> class depth
refit_stride = 8            # 2048 envs + 全身模板 → 175 ms/tick
update_interval = 10        # 可选：5 Hz depth，更贴近真机
```

### 建议打开的噪声增强

```python
# h1_loco_config.py -> class depth
rand_position = True
dis_noise = 0.03
gaussian_noise = True
gaussian_noise_std = 0.02
gaussian_filter = True
gaussian_filter_sigma = 1.0
```

### 确认无需修改

| 项 | 当前值 | 状态 |
|---|---|---|
| `warp_camera` | `True` | OK |
| `use_camera` | `False` | OK |
| `enable_self_occlusion` | `True` | OK |
| `robot_geom_module` | `"legged_gym.utils.h1_geom"` | OK |
| `add_body_mask` | `False` | OK（代码已不读） |

### 启动检查

```bash
cd /root/gpufree-data/workspace/more
python legged_gym/scripts/train.py
```

应看到：
```
[LeggedRobot] per-env self-occlusion BVH initialised: 2048 envs x 240 verts x 432 tris (links: [...], refit_stride=8)
```

---

## 9. 文件改动汇总

### 新增文件

| 文件 | 作用 |
|---|---|
| `legged_gym/utils/robot_geom.py` | 通用几何工具（LinkGeom / box / capsule / build_robot_template） |
| `legged_gym/utils/h1_geom.py` | H1 全身 12-link 简化几何表 |
| `legged_gym/scripts/verify_self_occlusion_bvh.py` | 独立 BVH 验证 + 性能基准脚本 |
| `resources/robots/h1/h1_camera.xml` | 带深度相机的 H1 MuJoCo 模型 |
| `resources/robots/h1/scene_camera.xml` | 含台阶的 MuJoCo 场景 |
| `deploy_camera/deploy_mujoco/deploy_mujoco_h1_camera.py` | MuJoCo 部署脚本 |
| `deploy_camera/deploy_mujoco/configs/h1_camera.yaml` | MuJoCo 部署配置 |
| `deploy_camera/deploy_real/deploy_real_h1_camera.py` | 实机部署脚本 |
| `deploy_camera/deploy_real/configs/h1_camera.yaml` | 实机部署配置 |
| `deploy_camera/README_h1_camera.md` | 部署文档（中文） |
| `docs/h1_camera_exploration.md` | 本文档 |

### 修改文件

| 文件 | 改动摘要 |
|---|---|
| `legged_gym/utils/warp_render_v3.py` | 重写 renderer：per-env BVH + dual ray-cast + 移除 pytorch3d 依赖 |
| `legged_gym/envs/base/legged_robot.py` | 集成 BVH init/update + 删除 body_mask 分支 + 缓存 body_names |
| `legged_gym/envs/base/legged_robot_config.py` | 新增 `enable_self_occlusion` / `robot_geom_module` / `refit_stride` |
| `legged_gym/envs/h1_loco/h1_loco_config.py` | H1 启用 self-occlusion 三个字段 |
| `legged_gym/envs/g1_loco/g1_16dof_moe_residual_env.py` | 删除 body_masks np.load |
| `rsl_rl/rsl_rl/runners/moe_residual_on_policy_runner_multi.py` | 删除 iter>30000 add_body_mask 切换 |

---

## 10. 待办 / 后续方向

### 高优先级

- [ ] **按新内参/全帧噪声流程重新训练并导出策略**：旧 policy 仍来自旧深度分布，重训后再同步 MuJoCo/实机 `policy_path`。
- [ ] **部署侧相机参数对齐**：MuJoCo 需要和训练内参的等效视场对齐；实机启动时应读取 RealSense profile 内参并和训练矩阵做日志/阈值校验。
- [ ] **自遮挡策略路线确认**：当前 H1 训练仍关闭 self-occlusion；如后续重训包含自遮挡，再把 MuJoCo `hide_robot_in_depth` 切到 `false`。

### 中优先级

- [ ] 实现 `random_cam_delay`：在 `warp_update_depth_buffer` 末尾给 buffer
  滚动加随机偏移（模拟 RealSense 帧延迟 jitter）。
- [ ] 考虑把 `update_interval` 从 5 调到 10（5 Hz depth），更贴近真机帧率。
- [ ] G1 复刻：提供 `g1_geom.py` 并设置 `robot_geom_module`，让 G1 也能用
  per-env BVH 替代 body_mask 数据集。
- [ ] 优化 refit 性能：探索 warp batched BVH refit 或 CUDA graph capture。

### 低优先级 / 长期

- [ ] Depth 引入 critic：让 value function 也接收深度信息，提高避障任务的
  sample efficiency。
- [ ] 二阶段 residual 训练与 per-env BVH 的兼容性验证（应该兼容，但未测试）。
- [ ] 简化几何参数精调：对比 STL 减面和手工 capsule 在渲染质量上的差异。

---

## 14. 2026-04-27 更新：480x640 全帧 + 底部居中裁剪

当前训练与部署的深度输入逻辑已改为：相机按 RealSense 深度流尺寸采集/渲染 **480x640 (H, W)** 全帧，再直接取底部居中的 **48x64 (H, W)** 裁剪输入策略网络；不再使用 resize 下采样。

裁剪窗口为默认 `bottom_margin=0`：rows `[432:480]`、cols `[288:352]`。如果真机安装或视野需要微调，只改部署/训练配置里的 `depth_crop_bottom_margin`，保持训练和部署一致。

H1 相机外参和内参随机化也同步收紧：`camera_parent_body="torso_link"`、`position=[0.10848, 0.01750, 0.69317]`、`y_angle=[50.8, 50.8]`、`x_angle=z_angle=[0, 0]`，并使用实测 640x480 pinhole 内参矩阵。`fovy_range=[58, 58]` 仅保留为兼容字段，不再驱动 H1 Warp ray。

### 2026-04-27 噪声设置更新

训练侧在保持 H1 相机内外参固定的前提下，已打开适度深度传感器噪声：`dis_noise=0.02`、`gaussian_noise=True`、`gaussian_noise_std=0.015`、`gaussian_filter=True`、`gaussian_filter_kernel=[3]`、`gaussian_filter_sigma=0.8`。相机位置/FOV/姿态随机化仍关闭，`random_cam_delay` 仍关闭。

---

## 15. 2026-04-27 更新：Warp 显式内参与全帧噪声后裁剪

这次修复关闭了 Warp depth renderer 里旧的 `fovy_dist_offset = 1 / tan(fovy/2) - 1` 射线生成路径。旧公式把横纵方向都间接绑到 FOV 偏移，在 480x640 这种非正方形全帧下会让实际 VFOV/HFOV 偏离 D435i 标定值。

H1 训练现在显式使用 RealSense 深度流 640x480 内参矩阵：

```text
[[384.77294921875, 0.0, 324.17236328125],
 [0.0, 384.77294921875, 236.48226928710938],
 [0.0, 0.0, 1.0]]
```

Warp 每个像素 `(u, v)` 的相机系 ray 改为 pinhole 形式：`normalize([1, -(u-cx)/fx, -(v-cy)/fy])`，平面深度仍沿用现有的 `d_fwd = rd_cam[0]` 写法。`fovy_range=[58,58]` 现在只保留作兼容/文档字段，不再驱动 H1 Warp ray。

训练侧深度处理顺序也改成全帧优先：先 render 得到 `(B,480,640)`，在完整图上加 per-pixel Gaussian noise、per-env distance bias、clip/normalize，再用 replicate padding 做 Gaussian filter，最后才取底部居中裁剪 `(B,48,64)` 写入 depth buffer。默认裁剪窗口保持 rows `[432:480]`、cols `[288:352]`。
