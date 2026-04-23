"""H1 + RealSense D435I depth-camera policy deployment.

Policy interface (policy_depth_1.pt):
    actions = policy(obs[1,39], history[1,10,39], depth[1,2,48,64])

Depth spatial size: 宽×高 = 64×48（与 640×480 同 4:3）。PyTorch 默认与训练文档一致，
使用 NCHW 且最后两维为 **H×W**，即 ``[..., 48, 64]``（高 48、宽 64）。若导出模型
实际为 ``[..., 64, 48]``，在 YAML 中设置 ``depth_tensor_layout: WH``。

obs layout (39-dim):
    [0:3]   cmd * cmd_scale          (velocity command)
    [3:6]   ang_vel * ang_vel_scale   (body angular velocity)
    [6:9]   projected_gravity
    [9:19]  (dof_pos - default) * dof_pos_scale
    [19:29] dof_vel * dof_vel_scale
    [29:39] last_action

Joint order (policy 0-9 -> SDK motor):
    0 left_hip_yaw->7   1 left_hip_roll->3   2 left_hip_pitch->4
    3 left_knee->5       4 left_ankle->10
    5 right_hip_yaw->8   6 right_hip_roll->0  7 right_hip_pitch->1
    8 right_knee->2      9 right_ankle->11
"""

import sys
from pathlib import Path

# 保证从任意工作目录启动时，能导入同目录下的 config_camera / common
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

sys.path.insert(0, "/usr/lib/python3/dist-packages/pyrealsense2")

import threading
import numpy as np
import time
import torch
import cv2

import pyrealsense2 as rs

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_go
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config_camera import CameraConfig


# ---------------------------------------------------------------------------
# RealSense depth capture (background thread)
# ---------------------------------------------------------------------------
class RealSenseDepth:
    """Continuously captures depth frames from a RealSense camera in a
    background daemon thread.  The main loop calls ``read()`` to get the
    latest normalised depth image.

    ``cv2.resize(..., (target_w, target_h))`` uses OpenCV *(width, height)*,
    so with defaults ``target_w=64``, ``target_h=48`` the numpy array has
    shape ``(48, 64)`` = *(rows=height, cols=width)*, i.e. **H×W** matching
    ``Tensor[B, 2, 48, 64]`` in the training doc.  Values in ``[-0.5, 0.5]``.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        target_h: int = 48,
        target_w: int = 64,
        near_clip: float = 0.0,
        far_clip: float = 2.0,
        rot90_k: int = 0,
    ):
        self.target_h = target_h
        self.target_w = target_w
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.rot90_k = rot90_k

        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._running = False

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self._pipeline.start(cfg)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[RealSenseDepth] started  scale={self._depth_scale:.6f}  "
              f"target={target_h}x{target_w}  far={far_clip}m  rot90_k={rot90_k}")

    def _loop(self):
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                continue
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            meters = raw * self._depth_scale

            invalid_mask = raw == 0
            meters[invalid_mask] = self.far_clip

            if self.rot90_k != 0:
                meters = np.rot90(meters, k=self.rot90_k)

            # dsize = (width, height) => numpy shape (height, width) = (H, W)
            resized = cv2.resize(
                meters,
                (self.target_w, self.target_h),
                interpolation=cv2.INTER_AREA,
            )

            clipped = np.clip(resized, self.near_clip, self.far_clip)
            normalised = clipped / self.far_clip - 0.5

            with self._lock:
                self._latest = normalised

    def read(self) -> np.ndarray | None:
        """Return the latest normalised depth frame, or *None* if no frame
        has been captured yet."""
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._pipeline.stop()
        print("[RealSenseDepth] stopped")


# ---------------------------------------------------------------------------
# Camera-aware controller
# ---------------------------------------------------------------------------
class CameraController:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        self.policy = torch.jit.load(config.policy_path)
        self.policy.eval()

        num_actions = config.num_actions
        num_obs = config.num_obs

        self.qj = np.zeros(num_actions, dtype=np.float32)
        self.dqj = np.zeros(num_actions, dtype=np.float32)
        self.action = np.zeros(num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.counter = 0

        self.trajectory_history = torch.zeros(
            1, config.obs_history_len, num_obs, dtype=torch.float32
        )

        self.depth_buffer = torch.zeros(
            1,
            config.depth_buffer_len,
            config.depth_height,
            config.depth_width,
            dtype=torch.float32,
        )

        self.realsense = RealSenseDepth(
            width=640,
            height=480,
            fps=30,
            target_h=config.depth_height,
            target_w=config.depth_width,
            near_clip=config.depth_near_clip,
            far_clip=config.depth_far_clip,
            rot90_k=config.depth_rot90_k,
        )

        # H1 uses the "go" msg type
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        self.wait_for_low_state()
        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    # ---- DDS callbacks ----------------------------------------------------

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdGo):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    # ---- State-machine helpers -------------------------------------------

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate(
            (self.config.default_angles, self.config.arm_waist_target), axis=0
        )
        dof_size = len(dof_idx)

        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = (
                    init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                )
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    # ---- Main control loop ------------------------------------------------

    def run(self):
        self.counter += 1

        # -- Read joint states (policy joint order via leg_joint2motor_idx) --
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[
                self.config.leg_joint2motor_idx[i]
            ].q
            self.dqj[i] = self.low_state.motor_state[
                self.config.leg_joint2motor_idx[i]
            ].dq

        # -- IMU (torso -> pelvis transform for H1) --
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[
                self.config.arm_waist_joint2motor_idx[0]
            ].q
            waist_yaw_omega = self.low_state.motor_state[
                self.config.arm_waist_joint2motor_idx[0]
            ].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel,
            )

        gravity_orientation = get_gravity_orientation(quat)

        # -- Scale sensor readings --
        qj_obs = (self.qj - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = self.dqj * self.config.dof_vel_scale
        ang_vel_scaled = ang_vel.flatten() * self.config.ang_vel_scale

        # -- Joystick -> velocity command --
        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        # -- Build obs (39-dim) --
        # ORDER: cmd | ang_vel | gravity | dof_pos | dof_vel | last_action
        n = self.config.num_actions  # 10
        self.obs[0:3] = self.cmd * self.config.max_cmd * self.config.cmd_scale
        self.obs[3:6] = ang_vel_scaled
        self.obs[6:9] = gravity_orientation
        self.obs[9 : 9 + n] = qj_obs
        self.obs[9 + n : 9 + 2 * n] = dqj_obs
        self.obs[9 + 2 * n : 9 + 3 * n] = self.action

        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0).float()

        # -- Update trajectory history (rolling window, newest at end) --
        self.trajectory_history = torch.cat(
            [self.trajectory_history[:, 1:, :], obs_tensor.unsqueeze(1)], dim=1
        )

        # -- Update depth buffer every cam_update_interval steps --
        if self.counter % self.config.cam_update_interval == 0:
            depth_frame = self.realsense.read()
            if depth_frame is not None:
                depth_tensor = (
                    torch.from_numpy(depth_frame).unsqueeze(0).unsqueeze(0).float()
                )
                self.depth_buffer = torch.cat(
                    [self.depth_buffer[:, 1:, :, :], depth_tensor], dim=1
                )

        # -- Policy forward: oldest 2 of 3 buffered depth frames --
        # 内部缓冲为 [B, T, H, W] = [1, 3, 48, 64]，对应分辨率「宽64×高48」
        depth_input = self.depth_buffer[:, :2, :, :]
        layout = getattr(self.config, "depth_tensor_layout", "HW").upper()
        if layout not in ("HW", "WH"):
            raise ValueError(
                f"depth_tensor_layout 应为 HW 或 WH，收到: {self.config.depth_tensor_layout!r}"
            )
        if layout == "WH":
            # JIT 若约定最后两维为 (W, H) = (64, 48)，与训练文档的 (H, W) 相反时启用
            depth_input = depth_input.transpose(2, 3).contiguous()

        with torch.no_grad():
            action_tensor = self.policy(
                obs_tensor, self.trajectory_history, depth_input
            )
        self.action = action_tensor.detach().numpy().squeeze()

        # -- Action -> target joint positions --
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # -- Build and send low-level command --
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="H1 depth-camera policy deployment with RealSense D435I"
    )
    parser.add_argument("net", type=str, help="network interface (e.g. enp3s0)")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="h1_camera.yaml",
        help="config file name in the configs folder",
    )
    args = parser.parse_args()

    config_path = _PROJECT_ROOT / "configs" / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    config = CameraConfig(config_path)

    ChannelFactoryInitialize(0, args.net)

    controller = CameraController(config)

    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    print("Entering policy loop (depth-camera mode)...")
    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    controller.realsense.stop()
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
