# 宇树 H1 · RealSense D435I 启动与实时查看指南

本机器人：**Unitree H1**
相机：**Intel RealSense D435I** (SN `231522071694`, FW `05.15.01.55`)
挂载 PC：**H1 PC2** (`192.168.123.162`, 账号 `unitree` / `Unitree0408`)
SDK：**librealsense2 v2.53.1**（源码编译，装于 `/usr/local`）

---

## 0. 连接到机器人

本机 `192.168.123.222` 与机器人同网段（`eno1`）。SSH 直连：

```bash
ssh unitree@192.168.123.162
# 密码: Unitree0408
```

若要批处理免交互：

```bash
sudo apt install -y sshpass     # 若未装
sshpass -p 'Unitree0408' ssh -o StrictHostKeyChecking=no unitree@192.168.123.162 '<命令>'
```

---

## 1. 硬件检查（每次使用前）

```bash
# 1) USB 层是否看到相机（VID 8086）
lsusb | grep 8086
# 预期: Bus 00x Device 0xx: ID 8086:0b3a Intel Corp. Intel(R) RealSense(TM) Depth Camera 435i

# 2) V4L2 节点
ls /dev/video*
# 预期: /dev/video0 ~ /dev/video5

# 3) librealsense 枚举
rs-enumerate-devices -s
# 预期:
#   Intel RealSense D435I   231522071694   05.15.01.55
```

如果 `lsusb` 看不到 `8086`，相机**硬件层**没被识别。先排除：

- 换一个 **蓝色/红色内芯的 USB 3 口**
- 必须用 RealSense 原装 **USB 3 数据线**（不是充电线）
- 拔插一次，等 3-5 秒枚举

---

## 2. 启动方式（三选一）

### 2.1 快速 30 帧抓拍（验证 SDK 工作）

脚本已部署在机器人的 `/tmp/start_realsense.py`，系统 Python 3.10 直接跑：

```bash
/usr/bin/python3 /tmp/start_realsense.py
```

预期输出：

```
[lib] pyrealsense2 loaded from: /usr/lib/python3/dist-packages/pyrealsense2/pyrealsense2.cpython-310-x86_64-linux-gnu.so
[enum] devices found: 1
  [0] Intel RealSense D435I SN=231522071694 FW=05.15.01.55 USB=3.2
[start] streaming from Intel RealSense D435I (IMU=True)
[frame 01] color=640x480 depth=640x480 center=2.072m ts=...
...
[ok] captured 30 frames in 1.43s (~21.0 fps)
[stop] pipeline stopped cleanly
```

### 2.2 MJPEG 实时流（浏览器观看，推荐）

脚本已部署在机器人的 `/tmp/rs_stream.py`。后台启动：

```bash
pkill -f rs_stream.py 2>/dev/null; sleep 1
nohup /usr/bin/python3 /tmp/rs_stream.py > /tmp/rs_stream.log 2>&1 &
```

在**本机浏览器**打开：

| 端点 | 内容 |
|---|---|
| [http://192.168.123.162:8080/](http://192.168.123.162:8080/) | 首页（默认显示 combined） |
| [http://192.168.123.162:8080/combined](http://192.168.123.162:8080/combined) | 左 RGB + 右彩色深度（1280×480） |
| [http://192.168.123.162:8080/color](http://192.168.123.162:8080/color) | 仅 RGB (640×480) |
| [http://192.168.123.162:8080/depth](http://192.168.123.162:8080/depth) | 仅彩色化深度 (640×480) |

深度色彩解释：红/黄=近，绿/青=中，蓝=远，黑=无效测距。

**查看日志**：

```bash
tail -f /tmp/rs_stream.log
```

**停止**：

```bash
pkill -f rs_stream.py
```

**调参**（编辑 `/tmp/rs_stream.py`）：

```python
PORT = 8080
WIDTH, HEIGHT, FPS = 640, 480, 30   # 可改 1280x720/30 或 848x480/60
JPEG_QUALITY = 70                    # 50~95，越高越清晰越占带宽
```

### 2.3 自己写 Python 代码（集成到工程）

两种 Python 环境可选，依赖不同：

| 环境 | Python | pyrealsense2 | 其它 |
|---|---|---|---|
| **系统 `/usr/bin/python3`** | 3.10.12 | **v2.53.1**（匹配本机 SDK） | ✅ cv2 4.5.4, numpy 1.26.4, PIL 9.0.1 |
| **conda `unitree`** | 3.8.18 | v2.55.1.6486（独立 pip 装的） | ❌ 无 cv2/PIL |

> **使用系统 Python 的注意事项**：
> `/usr/lib/python3/dist-packages/pyrealsense2/` 缺 `__init__.py`，直接 `import pyrealsense2` 会加载成空的 namespace package。必须先把 `.so` 所在目录前置到 `sys.path`：

```python
# 放在 import pyrealsense2 之前
import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages/pyrealsense2")
import pyrealsense2 as rs
```

最小骨架：

```python
import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages/pyrealsense2")
import pyrealsense2 as rs

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
# 可选 IMU：
cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
cfg.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, 200)

pipe.start(cfg)
try:
    while True:
        fs = pipe.wait_for_frames(timeout_ms=5000)
        color = fs.get_color_frame()
        depth = fs.get_depth_frame()
        # ... 你的处理 ...
finally:
    pipe.stop()
```

---

## 3. 客户端工具（可选）

| 工具 | 用途 | 命令 |
|---|---|---|
| `realsense-viewer` | GUI 全功能调试、标定、保存 bag | `realsense-viewer`（需 X 显示） |
| `rs-enumerate-devices` | 列设备 + FW 版本 | `rs-enumerate-devices -s` |
| `rs-depth` | ASCII 打印深度帧 | `rs-depth` |
| `rs-distance` | 打印画面中心距离 | `rs-distance` |
| `rs-data-collect` | 录制数据 | `rs-data-collect -t 10 -f data.bag` |
| `rs-fw-update` | 固件升级 | `rs-fw-update -l` 查询 |

---

## 4. 已知限制 / 坑

1. **H1 PC2 只有内网** (`192.168.123.0/24`)，默认无外网。升级固件 / 装 ROS `realsense2_camera` 需先把本机做 NAT 网关或把 .deb 离线传过去。
2. **系统 Python 的 pyrealsense2** 必须 `sys.path.insert(0, "/usr/lib/python3/dist-packages/pyrealsense2")`，否则 namespace package 加载不到 `.context()`。
3. **`/dev/video*` 最多 6 个节点**（D435I 有 color / depth-IR 双目 + IMU），属正常。
4. **USB 3 线是硬性要求** —— 2.0 线会被识别成低速设备，分辨率/帧率会被迫降到 480P@15 以下，甚至 `rs-enumerate-devices` 直接失败。
5. **一次只能有一个客户端打开设备**。如果 `realsense-viewer` 在跑，Python 脚本会打不开相机（反之亦然）。

---

## 5. 故障排查速查表

| 症状 | 原因 | 处理 |
|---|---|---|
| `lsusb` 无 `8086` | USB 物理层失败 | 换口/换线/重插 |
| `rs-enumerate-devices` → `No device detected` | 同上，或 udev 规则没装 | `ls /etc/udev/rules.d/99-realsense-libusb.rules` 确认存在 |
| Python 报 `AttributeError: module 'pyrealsense2' has no attribute 'context'` | 系统 Python 加载了 namespace 包 | 加 `sys.path.insert(0, "/usr/lib/python3/dist-packages/pyrealsense2")` |
| `wait_for_frames()` 超时 | 相机在别处被打开 / USB 2.0 | 关 realsense-viewer / 其它脚本；检查线 |
| 帧率远低于 30 fps | JPEG 编码 / SSH 占 CPU | 降 `JPEG_QUALITY` 或改小分辨率 |
| MJPEG 网页一直空白 | 浏览器不识别 `multipart/x-mixed-replace`（罕见） | 换 Chrome/Firefox；或 `ffplay http://.../color` |

---

## 6. 常用一键命令（本机执行）

```bash
# 启动 MJPEG 流
sshpass -p 'Unitree0408' ssh unitree@192.168.123.162 \
  'pkill -f rs_stream.py 2>/dev/null; sleep 1; nohup /usr/bin/python3 /tmp/rs_stream.py > /tmp/rs_stream.log 2>&1 &'

# 停止
sshpass -p 'Unitree0408' ssh unitree@192.168.123.162 'pkill -f rs_stream.py'

# 查看日志
sshpass -p 'Unitree0408' ssh unitree@192.168.123.162 'tail -f /tmp/rs_stream.log'

# 打开浏览器（Linux）
xdg-open http://192.168.123.162:8080/

# 用 ffplay 直接看（若不想开浏览器）
ffplay -fflags nobuffer -flags low_delay http://192.168.123.162:8080/color
```

---

## 7. 文件位置速查

机器人侧：
- SDK 库：`/usr/local/lib/librealsense2.so.2.53.1`
- CLI 工具：`/usr/local/bin/rs-*`，`/usr/local/bin/realsense-viewer`
- Python 绑定：`/usr/lib/python3/dist-packages/pyrealsense2/`
- udev 规则：`/etc/udev/rules.d/99-realsense-libusb.rules`
- Viewer 配置：`~/.realsense-config.json`
- 抓拍脚本：`/tmp/start_realsense.py`
- MJPEG 脚本：`/tmp/rs_stream.py`
- MJPEG 日志：`/tmp/rs_stream.log`
