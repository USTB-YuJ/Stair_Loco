# H1 头部相机（深度图）启用与部署集成指南

本文档基于对 [unitreerobotics](https://github.com/unitreerobotics) GitHub 组织全部相关仓库与官方文档的调研，整理 Unitree H1 / H1-2 头部相机的硬件信息、可用的官方/社区驱动方案、以及如何在 `LeggedLabDeploy` 这套部署代码里使用深度图。

---

## 1. H1 头部相机硬件事实（先看清楚再动手）

| 项 | H1 (V1) | H1-2 |
|---|---|---|
| 头部 LiDAR | Livox MID-360（IP `192.168.123.120`） | Livox MID-360 |
| 头部相机 | Intel RealSense **D435i**（USB 3.x 直连） | Intel RealSense **D435**（部分批次为 D435i） |
| RGB | 1920×1080@30 | 同上 |
| 深度 | 1280×720@90，理想范围 0.3m–3m，FOV 87°×58° | 同上 |
| IMU | D435i 内置 BMI055（D435 没有） | 视型号 |

参考来源：
- [Unitree H1 / H1-2 官方/MYBOTSHOP 文档](https://www.docs.quadruped.de/projects/h1/html/h1_overview.html)
- [Intel RealSense D435i 数据手册](https://www.intelrealsense.com/depth-camera-d435i/)

> 关键事实：H1 头部相机就是**一台标准的 USB RealSense D435/D435i**，并不是 Unitree 自研接口设备。这意味着任何 RealSense SDK / ROS 驱动都能用，只是物理上挂在机器人头上的 PC2 上。

---

## 2. H1 上的计算单元拓扑（决定相机访问方式）

H1 内部包含两到三台 PC：

| PC | IP | 用途 | 是否能用相机 |
|---|---|---|---|
| **PC1**（运动控制单元） | `192.168.123.162`/`163` | Unitree 内部运动控制，**用户不可访问** | 否 |
| **PC2**（开发计算单元） | `192.168.123.164` | 用户开发 PC（默认账号 `unitree` / 密码 `Unitree0408`） | **是**：D435i USB 直接挂这里 |
| PC3（可选） | `192.168.123.165` | 选配 Jetson 等扩展计算 | 视配置 |

参考来源：[H1 Network Interface 文档](https://www.docs.quadruped.de/projects/h1/html/interface.html)

**重要结论**：
- 头部 D435i 通过 USB 物理连接在 PC2 上。要拿深度图，**必须在 PC2 上运行驱动程序**。
- 你部署运行 `deploy.py` 的电脑（控制电脑）通过以太网连接到 PC2，需要从 PC2 把图像通过网络转发过来。
- 不能像访问 LiDAR 一样直接“连过去就有图像 topic”，因为 Unitree DDS 默认**不发布**头部相机数据流。

---

## 3. unitreerobotics 官方组织里的相关仓库（按相关性排序）

下表是我从 [unitreerobotics 主页](https://github.com/unitreerobotics) 全量仓库中筛出来与“H1 + 头部相机 + 部署”相关的项目：

| 仓库 | 用途 | 是否能直接用于 H1 头部深度图 |
|---|---|---|
| [`teleimager`](https://github.com/unitreerobotics/teleimager) | 官方图像服务，支持 UVC / OpenCV / **RealSense**，通过 ZeroMQ 或 WebRTC 推流 | 可改造，目前主要用于 **RGB 远程操作**，深度需要自行扩展 |
| [`xr_teleoperate`](https://github.com/unitreerobotics/xr_teleoperate)（原 `avp_teleoperate`） | XR 远程操作主框架，调用 `teleimager` 取头部相机 | 主要用 RGB（VR 视图），但 README 明确说明了**在 PC2 上启动 image_server 的标准流程** |
| [`unitree_sdk2_python`](https://github.com/unitreerobotics/unitree_sdk2_python) | DDS Python 接口 | **不支持** H1 头部相机；其 `front_camera` 例子是给 Go2 用的 |
| [`unitree_sdk2`](https://github.com/unitreerobotics/unitree_sdk2) | C++ 主 SDK | 同上，无头部相机接口 |
| [`unitree_ros`](https://github.com/unitreerobotics/unitree_ros) | ROS1 仿真包 | 不直接处理实物相机；issue #71 官方明确回复“用 `realsense-ros` 即可”|
| [`unitree_ros2`](https://github.com/unitreerobotics/unitree_ros2) | ROS2 + Go2/B2 | 不针对 H1 相机 |
| MYBOTSHOP H1 集成（[h1_depth_camera](https://www.docs.quadruped.de/projects/h1/html/h1_2_e.html)） | 第三方 ROS2 启动包 | 可参考其 `realsense.launch.py` 配置 |

总结：**Unitree 官方没有提供“深度图开箱即用”的封装库**。所有路径最终都落到“在 PC2 上跑 RealSense 驱动 + 把数据转发到控制电脑”。

---

## 4. H1 头部深度图的 3 种可行方案（推荐 → 备选）

### 方案 A（推荐，开发量最小）：在 PC2 上跑 `pyrealsense2`，自写一个轻量 ZMQ 推流脚本

- **优点**
  - 完全控制深度数据格式（uint16 z16、对齐到 RGB、缩放等）
  - 无 ROS 依赖，与 `LeggedLabDeploy` 现有 `deploy.py`（pure Python + `unitree_sdk2_python`）风格统一
  - 控制电脑用 ZMQ 订阅，pull/sub 都行
- **缺点**
  - 自己维护协议、序列化（建议直接 `numpy + ZMQ` 二进制传输）

我在本文 §6 给出了完整可运行的最小代码。

### 方案 B（ROS2 用户）：在 PC2 上跑官方 `realsense2_camera` ROS2 包

- 直接 `apt install ros-humble-realsense2-camera` 然后：
  ```bash
  ros2 launch realsense2_camera rs_launch.py \
       enable_depth:=true enable_color:=true align_depth.enable:=true
  ```
- 控制电脑上订阅 `/camera/aligned_depth_to_color/image_raw`
- 适合你已经有 ROS2 工作链的情况
- 注意：`align_depth:=true` 关键，否则深度和彩色像素不对齐

参考：[realsense-ros 官方仓库](https://github.com/IntelRealSense/realsense-ros)

### 方案 C（VR 远程操作场景）：复用 `teleimager` + 扩展深度通道

- 适用于你已经在用 `xr_teleoperate` 做 VR 操作、想顺便把深度也回传的场景
- 当前 `teleimager` 的 RealSense 模块默认推 BGR，要拿深度需要稍微修改它的发布逻辑，把 `depth_frame` 也加进 ZeroMQ topic
- 改动点不大但属于 fork 维护，长期不便
- 参考代码位置：[`teleimager/image_server.py`](https://github.com/unitreerobotics/teleimager/blob/main/image_server.py)

---

## 5. 部署前的软硬件准备（PC2 端）

### 5.1 检查相机能识别

SSH 登 PC2：
```bash
ssh unitree@192.168.123.164   # 默认密码 Unitree0408
```

确认设备：
```bash
lsusb | grep RealSense
# 期望看到 Intel(R) RealSense(TM) D435 / D435i
```

### 5.2 安装 librealsense + pyrealsense2

PC2 一般是 x86_64 (i5/i7) 或 ARM64 (Jetson Orin NX，G1/H1-2 可选)。对应安装：

```bash
# x86_64（H1 默认 PC2）
sudo apt update
sudo apt install -y librealsense2-dkms librealsense2-utils \
                    librealsense2-dev librealsense2-dbg
pip3 install pyrealsense2 numpy pyzmq opencv-python
```

```bash
# Jetson (ARM64)：用 NVIDIA 提供的源码编译，或 conda 安装
# 详见 https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
```

### 5.3 验证

```bash
realsense-viewer
# 能看到深度+彩色实时画面就 OK
```

如果识别成 `D430i` 而不是 `D435i`，通常是机器人内部 USB 接口接触不良（参考 [librealsense issue #14739](https://github.com/IntelRealSense/librealsense/issues/14739)）。

---

## 6. 方案 A 的最小可运行实现

### 6.1 PC2 端：`h1_camera_server.py`

放在 PC2 的 `~/h1_camera_server/` 下，开机自启或手动跑。

```python
"""H1 head camera (D435i) ZMQ depth+color publisher.

Run on PC2 (192.168.123.164). Publishes:
  - depth (uint16, mm)  on tcp://*:5566 topic 'depth'
  - color (bgr8)        on tcp://*:5566 topic 'color'
"""
import time
import struct

import numpy as np
import pyrealsense2 as rs
import zmq

WIDTH, HEIGHT, FPS = 640, 480, 30
ZMQ_BIND = "tcp://*:5566"


def make_pipeline():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(cfg)

    # Align depth to color so they share the same intrinsics/coordinates
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # meters per LSB
    return pipeline, align, depth_scale


def pack_frame(topic: bytes, frame: np.ndarray, ts: float) -> list[bytes]:
    """Pack a frame into a multipart ZMQ message."""
    h, w = frame.shape[:2]
    c = 1 if frame.ndim == 2 else frame.shape[2]
    header = struct.pack("dIIIB", ts, w, h, c,
                         {np.dtype("uint16"): 0,
                          np.dtype("uint8"): 1}[frame.dtype])
    return [topic, header, frame.tobytes()]


def main():
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.set_hwm(2)
    pub.bind(ZMQ_BIND)

    pipeline, align, depth_scale = make_pipeline()
    print(f"[INFO] depth_scale = {depth_scale} m/LSB; serving on {ZMQ_BIND}")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            ts = time.time()
            depth_np = np.asanyarray(depth.get_data())   # uint16, mm equivalent
            color_np = np.asanyarray(color.get_data())   # bgr8

            pub.send_multipart(pack_frame(b"depth", depth_np, ts))
            pub.send_multipart(pack_frame(b"color", color_np, ts))
    finally:
        pipeline.stop()
        pub.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
```

启动：
```bash
python3 ~/h1_camera_server/h1_camera_server.py
```

> 说明
> - `align(rs.stream.color)`：让 depth 像素和 color 像素一一对应，便于在策略里直接索引。
> - 序列化用最简单的 `struct + tobytes`：避免 cv2 编码带来的延迟，同时保留 uint16 深度精度。

### 6.2 控制电脑端：`h1_camera_client.py`（可作为模块给 deploy 用）

```python
"""H1 head camera ZMQ subscriber for use inside deploy.py."""
import struct
import threading
import time
from typing import Optional

import numpy as np
import zmq


class H1HeadCamera:
    """Latest-frame cache for color+depth from PC2."""

    def __init__(self, host: str = "192.168.123.164", port: int = 5566):
        self.addr = f"tcp://{host}:{port}"
        self._ctx = zmq.Context.instance()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.set_hwm(2)
        self._sub.setsockopt(zmq.CONFLATE, 0)  # we drain manually
        self._sub.connect(self.addr)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"depth")
        self._sub.setsockopt(zmq.SUBSCRIBE, b"color")

        self._lock = threading.Lock()
        self._latest = {"depth": None, "color": None}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                topic, header, payload = self._sub.recv_multipart()
            except zmq.error.ContextTerminated:
                return
            ts, w, h, c, dtype_id = struct.unpack("dIIIB", header)
            dtype = {0: np.uint16, 1: np.uint8}[dtype_id]
            arr = np.frombuffer(payload, dtype=dtype)
            arr = arr.reshape(h, w) if c == 1 else arr.reshape(h, w, c)
            with self._lock:
                self._latest[topic.decode()] = (ts, arr)

    def get_depth(self, max_age_s: float = 0.2) -> Optional[np.ndarray]:
        """Return latest depth frame as float32 in meters, or None if stale."""
        with self._lock:
            entry = self._latest.get("depth")
        if entry is None:
            return None
        ts, depth_u16 = entry
        if time.time() - ts > max_age_s:
            return None
        # H1 D435i default: 1 LSB = 0.001 m => meters
        return depth_u16.astype(np.float32) * 0.001

    def get_color(self, max_age_s: float = 0.2) -> Optional[np.ndarray]:
        with self._lock:
            entry = self._latest.get("color")
        if entry is None:
            return None
        ts, img = entry
        if time.time() - ts > max_age_s:
            return None
        return img

    def close(self):
        self._stop.set()
        self._sub.close(0)
```

### 6.3 集成到 `LeggedLabDeploy/deploy.py`

下面只展示**接入点**，你可以按需粘贴：

```python
from h1_camera_client import H1HeadCamera   # 把 6.2 的代码放到工程根目录

class Controller:
    def __init__(self, config, net):
        ...
        # 启用头部深度相机（PC2 的相机服务必须在跑）
        self.head_cam = H1HeadCamera(host="192.168.123.164", port=5566)

    def run(self):
        # 已有的策略推理流程 ...

        # ===== 取深度图 =====
        depth_m = self.head_cam.get_depth(max_age_s=0.1)   # H, W float32, meters
        if depth_m is not None:
            # 例 1：拿正前方 1m × 1m 区域的最小距离作为“前方障碍距离”
            h, w = depth_m.shape
            roi = depth_m[h // 2 - 60: h // 2 + 60,
                          w // 2 - 80: w // 2 + 80]
            valid = roi[(roi > 0.2) & (roi < 5.0)]
            front_dist = float(valid.min()) if valid.size else 5.0

            # 例 2：把 depth 加进策略观测（需要训练时同步加入这一维）
            # observation = np.concatenate([observation, [front_dist]])

        # 然后照常发指令到机器人
        ...
```

> 注意
> - `deploy.py` 目前 100Hz/50Hz 控制频率，相机 30Hz。**不要在控制循环里阻塞等图**，用 6.2 里的后台线程缓存最新帧 + `get_depth()` 非阻塞读取。
> - 训练侧没有这维观测时，不要随便把深度灌进策略 → 维度对不上策略会直接挂。

---

## 7. 几个常见“坑”和对应建议

1. **D435i 被识别成 D430i**（无 RGB）：先 `realsense-viewer` 看实际型号，硬件 USB 没插紧或线松了。
2. **PC2 是 ARM (Jetson)**：`pip install pyrealsense2` 拿不到 wheel，需要源码编译 librealsense（带 `-DBUILD_PYTHON_BINDINGS=ON`）。
3. **网络带宽**：640×480 depth(uint16) + color(bgr8) ≈ `(640*480*2 + 640*480*3) * 30 ≈ 46 MB/s`。机器人内部千兆没问题，跨 WiFi 一定要降到 320×240 或 15Hz。
4. **深度对齐**：必须 `rs.align(rs.stream.color)`，否则 depth 像素索引和 color 不一致，做 ROI / 物体距离判断会偏。
5. **延时**：从相机到策略大约 30~80ms（取决于 USB + ZMQ）。控制环用最新帧 + 时间戳过期检查（max_age_s）。
6. **不要走 unitree DDS 通道**：H1 DDS 没有发布头部相机 topic，不要花时间找这条路。
7. **多客户端**：方案 A 用 PUB/SUB，多个客户端可以同时订阅（比如 deploy 用 + 一个调试可视化用），互不干扰。

---

## 8. 推荐实施步骤

1. PC2 装 librealsense + pyrealsense2，跑 `realsense-viewer` 验证 D435i。
2. 在 PC2 部署 `h1_camera_server.py`（§6.1），先在 PC2 本机测：`ipython` 跑一段订阅代码确认能拿到帧。
3. 在控制电脑上跑 `h1_camera_client.py`（§6.2）单独测试；用 `cv2.imshow` 看一眼深度伪彩。
4. 在 `deploy.py` 里加一个**只读不参与控制**的相机线程，先验证网络稳定再考虑把深度灌进策略。
5. 真要把深度作为观测的一部分进策略，记得**训练侧也要改**：在 `legged_lab/envs/g1/g1_dwaq_env.py` 里加一个相同语义的观测，并保证维度一致。

---

## 9. 参考资料

### Unitree 官方
- [unitreerobotics/teleimager](https://github.com/unitreerobotics/teleimager) — 唯一官方涉及 RealSense 的图像服务
- [unitreerobotics/xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) — H1 头部相机的官方使用流程（VR 远程操作）
- [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) — H1 关节/IMU/遥控器接口（无头部相机）
- [Unitree H1 Network Interface 文档](https://www.docs.quadruped.de/projects/h1/html/interface.html)
- [Unitree H1 Component Overview](https://www.docs.quadruped.de/projects/h1/html/h1_overview.html)

### Intel RealSense
- [librealsense Python `align-depth2color` 示例](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py)
- [realsense-ros (ROS2 driver)](https://github.com/IntelRealSense/realsense-ros)
- [librealsense Jetson 安装指南](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md)

### 第三方 H1 ROS2 集成参考
- [MYBOTSHOP h1_2_e](https://www.docs.quadruped.de/projects/h1/html/h1_2_e.html) — 含 `h1_depth_camera/realsense.launch.py` 调用范式
