# H1 + RealSense 深度策略实机部署

独立部署工程：**Unitree H1** + **Intel RealSense D435I** 深度输入，运行 `policy_depth_1.pt` 类 JIT 策略。不依赖 `legged_gym` 或其它仓库目录，路径均以**本仓库根目录**为基准。

## 目录结构

```
<本仓库根>/
├── README.md
├── deploy_real_h1_camera.py   # 入口：DDS + RealSense + 策略
├── config_camera.py             # 配置加载；项目根 = 本文件所在目录
├── configs/
│   └── h1_camera.yaml           # 默认 YAML（含 policy_path 等）
├── weights/                     # 放置 policy_depth_1.pt（见下）
├── common/
│   ├── command_helper.py
│   ├── rotation_helper.py
│   └── remote_controller.py
└── docs/
    ├── README_h1_camera.md      # 策略张量、深度、训练侧对照（详细）
    └── realsense启动引导.md    # 相机环境、系统 Python 注意事项
```

## 策略权重

1. 将导出的 `policy_depth_1.pt` 放到 **`weights/policy_depth_1.pt`**（与默认 `configs/h1_camera.yaml` 中 `policy_path` 一致），或  
2. 在 `configs/h1_camera.yaml` 里修改 `policy_path`：
   - **相对路径**：相对本仓库根目录，例如 `weights/my_policy.pt`；
   - **绝对路径**：直接写完整路径；
   - 可选占位：`{PROJECT_ROOT}` 会替换为项目根目录（与 `config_camera.py` 中 `PROJECT_ROOT` 一致）。

## 依赖

```bash
pip install torch numpy pyyaml opencv-python scipy
# unitree_sdk2py、pyrealsense2 按官方方式安装到当前 Python 环境
```

在机器人系统 Python 下使用 RealSense 时，见 [docs/realsense启动引导.md](docs/realsense启动引导.md)；本脚本已包含常见的 `pyrealsense2` 路径注入。

## 启动

在**本仓库根目录**执行（任意工作目录也可，脚本会自行把项目根加入 `sys.path`）：

```bash
cd /path/to/deploy_real
python deploy_real_h1_camera.py <网卡名> [配置文件名]
```

示例：

```bash
python deploy_real_h1_camera.py enp3s0
python deploy_real_h1_camera.py enp3s0 h1_camera.yaml
```

- **网卡名**：与 H1 控制器通信的以太网接口，如 `enp3s0`。
- **配置文件名**：仅写文件名，实际路径为 `configs/<文件名>`；省略时默认为 `h1_camera.yaml`。

建议在 **连接 RealSense USB 且能访问机器人 DDS 的上位机** 上运行。

## 遥控器流程

1. 启动后 **零力矩**，确认安全后按 **START**。  
2. 约 2 s 插值到默认站姿；可缓慢下放吊具使双脚着地。  
3. 按 **A** 进入策略闭环。  
4. 摇杆：**左摇杆**前后/左右 → `vx` / `vy`（`lx` 已取反）；**右摇杆**左右 → `wz`（`rx` 已取反）。  
5. **SELECT** 或 **Ctrl+C** 退出 → 阻尼模式并停止 RealSense 线程。

## 策略接口速查

```text
action = policy(obs, history, depth)
```

| 张量 | 形状 | 说明 |
|------|------|------|
| `obs` | `[1, 39]` | `cmd(3) \| ω(3) \| gravity(3) \| q_rel(10) \| qd(10) \| last_action(10)` |
| `history` | `[1, 10, 39]` | 过去 10 步 `obs`，最新在 `[:, -1]` |
| `depth` | `[1, 2, 48, 64]` | 深度缓冲中最旧 2 帧，约 `[-0.5, 0.5]` |
| `action` | `[1, 10]` | 与 `default_angles` 及 `leg_joint2motor_idx` 的对应关系见下表 |

**策略关节顺序（第 `i` 维 → `leg_joint2motor_idx[i]`）**：

| i | 关节 | 默认 YAML 电机索引 |
|---|------|-------------------|
| 0 | left_hip_yaw | 7 |
| 1 | left_hip_roll | 3 |
| 2 | left_hip_pitch | 4 |
| 3 | left_knee | 5 |
| 4 | left_ankle | 10 |
| 5 | right_hip_yaw | 8 |
| 6 | right_hip_roll | 0 |
| 7 | right_hip_pitch | 1 |
| 8 | right_knee | 2 |
| 9 | right_ankle | 11 |

固件批次不同可能导致电机编号不一致，请在 `h1_camera.yaml` 中核对 `leg_joint2motor_idx`。

## 频率与深度

- 控制周期：`control_dt = 0.02` s（50 Hz）。  
- 深度入缓冲：每 `cam_update_interval`（默认 5）步，约 **10 Hz**。  
- 分辨率：配置里为 **宽 `depth_width` × 高 `depth_height`**（默认 64×48）。送入策略的张量默认形状为 `[1, 2, 48, 64]`，即 **先高后宽（H×W）**，与训练文档 `Tensor[B, 2, 48, 64]` 一致。若你的 JIT 最后两维是 `[64, 48]`，把 `depth_tensor_layout` 设为 `WH`。  
- 相机安装旋转：`depth_rot90_k`，见 [docs/README_h1_camera.md](docs/README_h1_camera.md)。

## 安全提示

本工程用于科研与联调，**不是**产品级安全认证控制器。请在有保护、人员远离足端的环境下使用；异常立即 **SELECT** 或断电。

## 延伸阅读

- [docs/README_h1_camera.md](docs/README_h1_camera.md)：观测/深度细节、与训练配置对照、自遮挡说明。  
- [docs/realsense启动引导.md](docs/realsense启动引导.md)：硬件与 SDK、`rs-enumerate-devices`、MJPEG 调试等。
