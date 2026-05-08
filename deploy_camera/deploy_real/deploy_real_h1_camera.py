"""Real-robot deployment for the H1 loco-with-depth policy.

Loads `policy_depth_1.pt` (exported by `PolicyExporterDepth` in
`legged_gym/utils/helpers.py`) and runs it on a physical Unitree H1 with
an Intel RealSense depth camera mounted on the pelvis.

The exported policy expects:
    actions = policy(
        obs:     Tensor[1, 39],          # current proprio observation
        history: Tensor[1, 10, 39],      # last 10 proprio observations
        depth:   Tensor[1, 2, 48, 64],   # 2-frame depth stack (H×W, 4:3)
    ) -> Tensor[1, 10]                   # action mean for the 10 leg joints

Joint order convention used in the obs/action vectors (matches IsaacGym DOF order
declared in `H1_Loco_Cfg.init_state.default_joint_angles`):
    0  left_hip_yaw     5  right_hip_yaw
    1  left_hip_roll    6  right_hip_roll
    2  left_hip_pitch   7  right_hip_pitch
    3  left_knee        8  right_knee
    4  left_ankle       9  right_ankle

`leg_joint2motor_idx` in the YAML maps these *policy* indices to the H1
hardware motor indices that the Unitree SDK expects.

Camera notes:
    - The training env feeds depth in [-0.5, 0.5] where -0.5 = near (0 m) and
      0.5 = far (2 m); we replicate that normalization here.
    - We keep a 3-frame chronological buffer (newest at the end) and feed
      `depth[:, :2, ...]` to the policy, mirroring `play.py`.
"""

import argparse
import threading
import time
from typing import Union

import cv2
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_,
    unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import (
    MotorMode, create_damping_cmd, create_zero_cmd, init_cmd_go, init_cmd_hg,
)
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from config import Config


# ---------------------------------------------------------------------------
# RealSense depth source
# ---------------------------------------------------------------------------
class RealSenseDepth:
    """Background-thread RealSense depth grabber.

    Produces a bottom-center H×W crop (default 48×64) from the full RealSense
    depth frame, then normalizes it to [-0.5, 0.5].  No resize/downsampling is
    applied before the policy input.
    """

    def __init__(self, near_clip=0.0, far_clip=2.0, out_size=(48, 64),
                 input_size=(480, 640), fps=30, rot90_k=0,
                 crop_bottom_margin=0):
        try:
            import pyrealsense2 as rs  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pyrealsense2 is required for the real-robot deployment. "
                "Install with `pip install pyrealsense2`."
            ) from e
        self.rs = __import__("pyrealsense2")
        self.near = float(near_clip)
        self.far = float(far_clip)
        self.out_h, self.out_w = int(out_size[0]), int(out_size[1])
        self.in_h, self.in_w = int(input_size[0]), int(input_size[1])
        self.fps = int(fps)
        self.crop_bottom_margin = int(crop_bottom_margin)
        # rot90_k matches the camera's physical orientation to the natural
        # "top = horizon, left = robot's left" image layout the policy was
        # trained on. For a RealSense mounted upright (USB port at the bottom),
        # k=0. If you mount it rotated 90 deg, set k=1 / -1 / 2 accordingly.
        self.rot90_k = int(rot90_k)

        self._pipeline = self.rs.pipeline()
        cfg = self.rs.config()
        cfg.enable_stream(self.rs.stream.depth, self.in_w, self.in_h,
                          self.rs.format.z16, self.fps)
        profile = self._pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())  # meters per unit

        self._lock = threading.Lock()
        self._latest = np.zeros((self.out_h, self.out_w), dtype=np.float32)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            while not self._stop.is_set():
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                raw = np.asanyarray(depth_frame.get_data())  # uint16, mm-units
                meters = raw.astype(np.float32) * self.depth_scale
                # invalid pixels (==0) -> far_clip so they're treated as "far"
                meters[meters <= 1e-3] = self.far
                if self.rot90_k:
                    meters = np.rot90(meters, k=self.rot90_k).copy()
                meters = self._crop_bottom_center(meters)
                meters = np.clip(meters, self.near, self.far)
                normalized = (meters - self.near) / (self.far - self.near) - 0.5
                with self._lock:
                    self._latest = normalized.astype(np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"[RealSenseDepth] capture thread stopped: {e}")

    def _crop_bottom_center(self, image: np.ndarray) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        top = src_h - self.crop_bottom_margin - self.out_h
        left = (src_w - self.out_w) // 2
        if top < 0 or left < 0:
            raise ValueError(
                f"Cannot crop {self.out_h}x{self.out_w} from RealSense frame "
                f"{src_h}x{src_w} with bottom_margin={self.crop_bottom_margin}."
            )
        return image[top:top + self.out_h, left:left + self.out_w]

    def read(self) -> np.ndarray:
        with self._lock:
            return self._latest.copy()

    def close(self):
        self._stop.set()
        try:
            self._pipeline.stop()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class CameraController:
    def __init__(self, config: Config, depth_source: RealSenseDepth):
        self.config = config
        self.depth_source = depth_source
        self.remote_controller = RemoteController()

        self.policy = torch.jit.load(config.policy_path)
        self.policy.eval()

        # ----- proprio + history state -----
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs_buf = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)

        self.trajectory_history = torch.zeros(
            1, config.obs_history_len, config.num_obs, dtype=torch.float32
        )
        dh, dw = int(config.depth_size[0]), int(config.depth_size[1])
        self._depth_h, self._depth_w = dh, dw
        self.depth_buffer = torch.zeros(
            1, config.depth_buffer_len, dh, dw, dtype=torch.float32
        )
        self.depth_initialized = False

        self.counter = 0
        self.cam_counter = 0

        # ----- Unitree DDS channels -----
        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
        else:
            raise ValueError(f"Invalid msg_type: {config.msg_type}")

        self.wait_for_low_state()
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        else:
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    # --------------------------- DDS callbacks ---------------------------
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    # --------------------------- Start-up flow ---------------------------
    def zero_torque_state(self):
        print("Enter zero torque state. Waiting for the START signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pose...")
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
                m = dof_idx[j]
                self.low_cmd.motor_cmd[m].q = init_dof_pos[j] * (1 - alpha) + default_pos[j] * alpha
                self.low_cmd.motor_cmd[m].qd = 0
                self.low_cmd.motor_cmd[m].kp = kps[j]
                self.low_cmd.motor_cmd[m].kd = kds[j]
                self.low_cmd.motor_cmd[m].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state. Waiting for the A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                m = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[m].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[m].qd = 0
                self.low_cmd.motor_cmd[m].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[m].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[m].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                m = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[m].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[m].qd = 0
                self.low_cmd.motor_cmd[m].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[m].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[m].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    # --------------------------- Main loop step --------------------------
    def run(self):
        self.counter += 1

        # Read joint state in *policy* order using the SDK index map.
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        quat = self.low_state.imu_state.quaternion  # wxyz
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        if self.config.imu_type == "torso":
            # H1's IMU is on the torso; rotate readings to pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat, imu_omega=ang_vel,
            )

        grav = get_gravity_orientation(quat)
        qj_n = (self.qj - self.config.default_angles) * self.config.dof_pos_scale
        dqj_n = self.dqj * self.config.dof_vel_scale
        omega_n = ang_vel * self.config.ang_vel_scale

        # ---- velocity command from joystick ----
        # Stick scaling: map [-1, 1] -> physical command range, then apply obs scaling.
        self.cmd[0] = self.remote_controller.ly * self.config.max_cmd[0]      # vx
        self.cmd[1] = -self.remote_controller.lx * self.config.max_cmd[1]     # vy
        self.cmd[2] = -self.remote_controller.rx * self.config.max_cmd[2]     # wz

        n = self.config.num_actions
        self.obs_buf[0:3] = self.cmd * self.config.cmd_scale
        self.obs_buf[3:6] = omega_n
        self.obs_buf[6:9] = grav
        self.obs_buf[9:9 + n] = qj_n
        self.obs_buf[9 + n:9 + 2 * n] = dqj_n
        self.obs_buf[9 + 2 * n:9 + 3 * n] = self.action

        obs_tensor = torch.from_numpy(self.obs_buf).float().unsqueeze(0)
        self.trajectory_history = torch.cat(
            [self.trajectory_history[:, 1:], obs_tensor.unsqueeze(1)], dim=1
        )

        # ---- depth refresh at 1/cam_update_interval of policy rate ----
        if self.cam_counter % self.config.cam_update_interval == 0:
            depth_np = self.depth_source.read()  # (H, W), already normalized
            depth_t = torch.from_numpy(depth_np).float()
            if not self.depth_initialized:
                self.depth_buffer = depth_t.expand(
                    1, self.config.depth_buffer_len, self._depth_h, self._depth_w
                ).clone()
                self.depth_initialized = True
            else:
                self.depth_buffer = torch.cat(
                    [self.depth_buffer[:, 1:],
                     depth_t.unsqueeze(0).unsqueeze(0)],
                    dim=1,
                )
        self.cam_counter += 1

        # The exported policy slices the FIRST 2 frames of the 3-frame buffer
        depth_in = self.depth_buffer[:, :2, ...]
        with torch.no_grad():
            action_t = self.policy(obs_tensor, self.trajectory_history, depth_in)
        self.action = action_t.detach().cpu().numpy().squeeze(0).astype(np.float32)

        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build leg motor command
        for i in range(len(self.config.leg_joint2motor_idx)):
            m = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[m].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[m].qd = 0
            self.low_cmd.motor_cmd[m].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[m].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[m].tau = 0

        # Hold arms / torso at fixed targets
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            m = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[m].q = float(self.config.arm_waist_target[i])
            self.low_cmd.motor_cmd[m].qd = 0
            self.low_cmd.motor_cmd[m].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[m].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[m].tau = 0

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


# ---------------------------------------------------------------------------
# Camera-aware Config wrapper
# ---------------------------------------------------------------------------
class CameraConfig(Config):
    """Extends `Config` with depth/history fields needed by the camera policy."""

    def __init__(self, file_path) -> None:
        super().__init__(file_path)
        import yaml
        with open(file_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.obs_history_len = int(cfg["obs_history_len"])
        self.depth_near_clip = float(cfg["depth_near_clip"])
        self.depth_far_clip = float(cfg["depth_far_clip"])
        self.depth_buffer_len = int(cfg["depth_buffer_len"])
        self.depth_size = tuple(cfg.get("depth_size", [48, 64]))  # (H, W)
        self.cam_update_interval = int(cfg["cam_update_interval"])
        self.realsense_input_size = tuple(cfg.get("realsense_input_size", [480, 640]))
        self.realsense_fps = int(cfg.get("realsense_fps", 30))
        self.depth_rot90_k = int(cfg.get("depth_rot90_k", 0))
        self.depth_crop_bottom_margin = int(cfg.get("depth_crop_bottom_margin", 0))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface, e.g. enp3s0")
    parser.add_argument("config", type=str, nargs="?", default="h1_camera.yaml",
                        help="config file in deploy_camera/deploy_real/configs/")
    args = parser.parse_args()

    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy_camera/deploy_real/configs/{args.config}"
    config = CameraConfig(config_path)

    ChannelFactoryInitialize(0, args.net)

    depth_source = RealSenseDepth(
        near_clip=config.depth_near_clip,
        far_clip=config.depth_far_clip,
        out_size=tuple(config.depth_size),
        input_size=config.realsense_input_size,
        fps=config.realsense_fps,
        rot90_k=config.depth_rot90_k,
        crop_bottom_margin=config.depth_crop_bottom_margin,
    )

    controller = CameraController(config, depth_source)
    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    print("Entering policy control loop. Press SELECT to exit.")
    try:
        while True:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        create_damping_cmd(controller.low_cmd)
        controller.send_cmd(controller.low_cmd)
        depth_source.close()
        print("Exit")


if __name__ == "__main__":
    main()
