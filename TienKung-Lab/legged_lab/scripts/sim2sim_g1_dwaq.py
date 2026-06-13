# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""
G1 DWAQ Sim2Sim 脚本 (带步态相位版本)
=====================================

将在 Isaac Lab 中训练的 G1 机器人 DWAQ 策略迁移到 MuJoCo 仿真环境中运行。

DWAQ (Deep Variational Autoencoder for Walking) 使用 β-VAE 从观测历史中学习
潜在表示，用于盲走 (blind walking)。

网络架构 (ActorCritic_DWAQ):
----------------------------
- 观测历史 (100 x 5 = 500 dim) -> VAE Encoder -> latent code (19 dim)
  - velocity (3 dim) + latent (16 dim) = 19 dim
- Actor 输入: current_obs (100) + latent_code (19) = 119 dim
- Actor 输出: 29 dim (actions)

观测结构 (100 维) - 带步态相位版本:
-----------------------------------
- ang_vel (3): 角速度 (body frame)
- projected_gravity (3): 投影重力
- command (3): 速度命令 [vx, vy, yaw_rate]
- joint_pos (29): 关节位置偏差 (当前 - 默认)
- joint_vel (29): 关节速度
- action (29): 上一步动作
- gait_phase (4): 步态相位 [sin_left, cos_left, sin_right, cos_right]

使用方法：
---------
python legged_lab/scripts/sim2sim_g1_dwaq.py --checkpoint <model.pt>

键盘控制 (小键盘)：
------------------
- 8/2: 前进/后退
- 4/6: 左移/右移  
- 7/9: 左转/右转
- 5: 停止
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
from pynput import keyboard


# ==================== 关节顺序定义 ====================
# MuJoCo XML 中的关节顺序 (29 DOF)
MUJOCO_DOF_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
]

# Isaac Lab 中的关节顺序 (按 URDF actuator 定义)
LAB_DOF_NAMES = [
    'left_hip_pitch_joint',
    'right_hip_pitch_joint',
    'waist_yaw_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'waist_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'waist_pitch_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_shoulder_roll_joint',
    'right_shoulder_roll_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
    'left_shoulder_yaw_joint',
    'right_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint'
]


# ==================== 网络定义 ====================
# 从 rsl_rl/modules/actor_critic_DWAQ.py 复制，确保兼容性

def get_activation(act_name: str) -> nn.Module | None:
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class ActorCritic_DWAQ(nn.Module):
    """Actor-Critic with DWAQ (Deep Variational Autoencoder for Walking) context encoder.
    
    The context encoder (β-VAE) infers velocity and latent state from observation history.
    Sim2Sim 版本只包含 Actor 和 Encoder 部分。
    """
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cenet_in_dim: int,
        cenet_out_dim: int,
        obs_dim: int,
        activation: str = "elu",
        init_noise_std: float = 1.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.activation = get_activation(activation)
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(num_actor_obs, 512),
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, num_actions)
        )

        # VAE Encoder
        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim, 128),
            self.activation,
            nn.Linear(128, 64),
            self.activation,
        )
        self.encode_mean_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_logvar_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)

        # Decoder (for completeness, not used in inference)
        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim, 64),
            self.activation,
            nn.Linear(64, 128),
            self.activation,
            nn.Linear(128, self.obs_dim)
        )

        print(f"[INFO] ActorCritic_DWAQ 初始化:")
        print(f"  - Actor 输入: {num_actor_obs} (obs + latent_code)")
        print(f"  - Encoder 输入: {cenet_in_dim} (obs_history)")
        print(f"  - Encoder 输出: {cenet_out_dim} (vel:3 + latent:{cenet_out_dim-3})")
        print(f"  - Actor 输出: {num_actions}")

    def cenet_forward(self, obs_history: torch.Tensor):
        """Forward pass through the context encoder (β-VAE).
        
        Args:
            obs_history: Flattened observation history [batch, history_len * obs_dim]
            
        Returns:
            code: Concatenated latent code [vel(3) + latent(16)] for actor
        """
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        # 推理时使用均值，不采样
        code = torch.cat((mean_vel, mean_latent), dim=-1)
        return code

    def act_inference(self, observations: torch.Tensor, obs_history: torch.Tensor):
        """Compute deterministic actions for inference.
        
        Args:
            observations: Current actor observations [batch, obs_dim]
            obs_history: Observation history for context encoder [batch, history_len * obs_dim]
            
        Returns:
            Mean actions (deterministic)
        """
        code = self.cenet_forward(obs_history)
        actor_input = torch.cat((code, observations), dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean


class G1DwaqSim2SimCfg:
    """G1 DWAQ Sim2Sim 配置类 - 无步态版本"""

    class sim:
        sim_duration = 100.0
        num_actions = 29
        num_obs_per_step = 100     # 3+3+3+29+29+29+4 = 100 (带步态相位)
        dwaq_obs_history_length = 5  # DWAQ 使用 5 帧历史
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25
        
        # DWAQ encoder output: velocity(3) + latent(16) = 19
        cenet_out_dim = 19

    class robot:
        # G1 初始高度
        init_height = 0.793
    
    class gait_phase:
        # 步态相位配置 (与训练配置一致)
        enable = True
        period = 0.8   # 步态周期 0.8s
        offset = 0.5   # 左右腿相位差 50%


class JointLimitMonitor:
    """Record sim2sim joint/torque limit violations and plot only failed joints."""

    def __init__(self, model: mujoco.MjModel, joint_names: list[str], log_root: str, tolerance: float = 1e-6):
        self.model = model
        self.joint_names = list(joint_names)
        self.log_root = log_root
        self.tolerance = tolerance
        self.num_joints = len(self.joint_names)

        self.joint_ids = self._find_joint_ids()
        self.actuator_ids = self._find_actuator_ids()
        self.joint_limits = self._read_joint_limits()
        self.torque_limits = self._read_torque_limits()

        self.times: list[float] = []
        self.qpos: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.torques: list[np.ndarray] = []

        self.qpos_over_max = np.zeros(self.num_joints, dtype=np.float64)
        self.target_over_max = np.zeros(self.num_joints, dtype=np.float64)
        self.torque_over_max = np.zeros(self.num_joints, dtype=np.float64)
        self.qpos_over_count = np.zeros(self.num_joints, dtype=np.int64)
        self.target_over_count = np.zeros(self.num_joints, dtype=np.int64)
        self.torque_over_count = np.zeros(self.num_joints, dtype=np.int64)

    def _find_joint_ids(self) -> list[int]:
        joint_ids = []
        for name in self.joint_names:
            joint_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
        return joint_ids

    def _find_actuator_ids(self) -> list[int]:
        actuator_ids = []
        for idx, name in enumerate(self.joint_names):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0 and self.joint_ids[idx] >= 0:
                matches = np.where(self.model.actuator_trnid[:, 0] == self.joint_ids[idx])[0]
                if len(matches) > 0:
                    actuator_id = int(matches[0])
            actuator_ids.append(actuator_id)
        return actuator_ids

    def _read_joint_limits(self) -> np.ndarray:
        limits = np.full((self.num_joints, 2), np.nan, dtype=np.float64)
        for idx, joint_id in enumerate(self.joint_ids):
            if joint_id < 0:
                print(f"[WARNING] Limit monitor could not find joint: {self.joint_names[idx]}")
                continue
            if hasattr(self.model, 'jnt_limited') and not self.model.jnt_limited[joint_id]:
                continue
            limits[idx] = self.model.jnt_range[joint_id]
        return limits

    def _read_torque_limits(self) -> np.ndarray:
        limits = np.full((self.num_joints, 2), np.nan, dtype=np.float64)
        for idx, actuator_id in enumerate(self.actuator_ids):
            if actuator_id < 0:
                print(f"[WARNING] Limit monitor could not find actuator for joint: {self.joint_names[idx]}")
                continue
            if hasattr(self.model, 'actuator_ctrllimited') and not self.model.actuator_ctrllimited[actuator_id]:
                continue
            limits[idx] = self.model.actuator_ctrlrange[actuator_id]
        return limits

    @staticmethod
    def _limit_over(values: np.ndarray, limits: np.ndarray) -> np.ndarray:
        over = np.zeros(values.shape, dtype=np.float64)
        valid = np.isfinite(limits[:, 0]) & np.isfinite(limits[:, 1])
        if np.any(valid):
            over[valid] = np.maximum(limits[valid, 0] - values[valid], values[valid] - limits[valid, 1])
            over = np.maximum(over, 0.0)
        return over

    @staticmethod
    def _safe_filename(name: str) -> str:
        return ''.join(c if c.isalnum() or c in '._-' else '_' for c in name)

    def record(self, time_s: float, qpos: np.ndarray, target_q: np.ndarray, tau: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64)[: self.num_joints].copy()
        target_q = np.asarray(target_q, dtype=np.float64)[: self.num_joints].copy()
        tau = np.asarray(tau, dtype=np.float64)[: self.num_joints].copy()

        self.times.append(float(time_s))
        self.qpos.append(qpos)
        self.targets.append(target_q)
        self.torques.append(tau)

        qpos_over = self._limit_over(qpos, self.joint_limits)
        target_over = self._limit_over(target_q, self.joint_limits)
        torque_over = self._limit_over(tau, self.torque_limits)

        self.qpos_over_max = np.maximum(self.qpos_over_max, qpos_over)
        self.target_over_max = np.maximum(self.target_over_max, target_over)
        self.torque_over_max = np.maximum(self.torque_over_max, torque_over)
        self.qpos_over_count += qpos_over > self.tolerance
        self.target_over_count += target_over > self.tolerance
        self.torque_over_count += torque_over > self.tolerance

    def _violated_indices(self) -> np.ndarray:
        violated = (
            (self.qpos_over_count > 0)
            | (self.target_over_count > 0)
            | (self.torque_over_count > 0)
        )
        return np.flatnonzero(violated)

    def _summary(self, violated_indices: np.ndarray) -> dict:
        joints = []
        for idx in violated_indices:
            joints.append(
                {
                    'joint': self.joint_names[idx],
                    'joint_limit': self.joint_limits[idx].tolist(),
                    'torque_limit': self.torque_limits[idx].tolist(),
                    'qpos_over_count': int(self.qpos_over_count[idx]),
                    'qpos_over_max': float(self.qpos_over_max[idx]),
                    'target_over_count': int(self.target_over_count[idx]),
                    'target_over_max': float(self.target_over_max[idx]),
                    'torque_over_count': int(self.torque_over_count[idx]),
                    'torque_over_max': float(self.torque_over_max[idx]),
                }
            )
        return {
            'num_samples': len(self.times),
            'num_violated_joints': int(len(violated_indices)),
            'violated_joints': joints,
        }

    def save(self) -> str | None:
        violated_indices = self._violated_indices()
        if len(violated_indices) == 0:
            print('[INFO] Limit monitor: no joint/torque limit violations; no plots saved')
            return None

        if not self.times:
            print('[WARNING] Limit monitor: violations were marked but no samples were recorded')
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(self.log_root, timestamp)
        os.makedirs(log_dir, exist_ok=True)

        time_arr = np.asarray(self.times)
        qpos_arr = np.asarray(self.qpos)
        target_arr = np.asarray(self.targets)
        torque_arr = np.asarray(self.torques)

        summary = self._summary(violated_indices)
        with open(os.path.join(log_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        np.savez_compressed(
            os.path.join(log_dir, 'limit_curves.npz'),
            time=time_arr,
            qpos=qpos_arr,
            target=target_arr,
            torque=torque_arr,
            joint_limits=self.joint_limits,
            torque_limits=self.torque_limits,
            joint_names=np.asarray(self.joint_names),
        )

        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        for idx in violated_indices:
            name = self.joint_names[idx]
            curve = np.column_stack(
                [
                    time_arr,
                    qpos_arr[:, idx],
                    target_arr[:, idx],
                    torque_arr[:, idx],
                ]
            )
            np.savetxt(
                os.path.join(log_dir, f'{self._safe_filename(name)}.csv'),
                curve,
                delimiter=',',
                header='time,qpos,target,torque',
                comments='',
            )

            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            axes[0].plot(time_arr, qpos_arr[:, idx], label='qpos', linewidth=1.2)
            axes[0].plot(time_arr, target_arr[:, idx], label='target', linewidth=1.0)
            lower, upper = self.joint_limits[idx]
            if np.isfinite(lower) and np.isfinite(upper):
                axes[0].axhline(lower, color='tab:red', linestyle='--', linewidth=1.0, label='joint limit')
                axes[0].axhline(upper, color='tab:red', linestyle='--', linewidth=1.0)
            axes[0].set_ylabel('joint pos (rad)')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc='best')

            axes[1].plot(time_arr, torque_arr[:, idx], label='torque', color='tab:green', linewidth=1.2)
            tau_lower, tau_upper = self.torque_limits[idx]
            if np.isfinite(tau_lower) and np.isfinite(tau_upper):
                axes[1].axhline(tau_lower, color='tab:red', linestyle='--', linewidth=1.0, label='torque limit')
                axes[1].axhline(tau_upper, color='tab:red', linestyle='--', linewidth=1.0)
            axes[1].set_xlabel('time (s)')
            axes[1].set_ylabel('torque (Nm)')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc='best')

            fig.suptitle(
                f"{name}: target_over={self.target_over_max[idx]:.4f}, "
                f"qpos_over={self.qpos_over_max[idx]:.4f}, "
                f"torque_over={self.torque_over_max[idx]:.4f}"
            )
            fig.tight_layout()
            fig.savefig(os.path.join(log_dir, f'{self._safe_filename(name)}.png'), dpi=150)
            plt.close(fig)

        print(f"[WARNING] Limit monitor saved {len(violated_indices)} violated joint plots to: {log_dir}")
        for item in summary['violated_joints']:
            print(
                f"  - {item['joint']}: "
                f"target_over_max={item['target_over_max']:.4f}, "
                f"qpos_over_max={item['qpos_over_max']:.4f}, "
                f"torque_over_max={item['torque_over_max']:.4f}"
            )
        return log_dir


class MujocoVideoRecorder:
    """Stream MuJoCo frames to MP4 and close the writer on exit."""

    def __init__(
        self,
        model: mujoco.MjModel,
        video_path: str,
        fps: float = 50.0,
        width: int = 640,
        height: int = 480,
        camera: str | None = None,
        follow_distance: float = 3.0,
        follow_height: float = 0.75,
        follow_yaw_offset: float = -45.0,
        follow_elevation: float = -28.0,
    ):
        self.model = model
        self.video_path = os.path.abspath(video_path)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        offscreen_width = int(getattr(self.model.vis.global_, "offwidth", self.width))
        offscreen_height = int(getattr(self.model.vis.global_, "offheight", self.height))
        if self.width > offscreen_width or self.height > offscreen_height:
            print(
                f"[WARNING] Requested video size {self.width}x{self.height} exceeds "
                f"MuJoCo offscreen framebuffer {offscreen_width}x{offscreen_height}; clamping."
            )
            self.width = min(self.width, offscreen_width)
            self.height = min(self.height, offscreen_height)
        camera_mode = "follow_side_rear" if camera in (None, "") else camera
        self.follow_camera = camera_mode in ("follow", "side_rear", "follow_side_rear")
        self.camera = None if self.follow_camera or camera_mode == "free" else camera_mode
        self.follow_distance = float(follow_distance)
        self.follow_height = float(follow_height)
        self.follow_yaw_offset = float(follow_yaw_offset)
        self.follow_elevation = float(follow_elevation)
        self.mjv_camera = mujoco.MjvCamera()
        self.mjv_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.next_frame_time = 0.0
        self.frame_interval = 1.0 / max(self.fps, 1e-6)
        self.frame_count = 0
        self.closed = False

        os.makedirs(os.path.dirname(self.video_path), exist_ok=True)

        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "Video recording requires imageio. Install imageio or run without --record-video."
            ) from exc

        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.writer = imageio.get_writer(self.video_path, fps=self.fps, macro_block_size=1)
        camera_desc = "follow_side_rear" if self.follow_camera else (self.camera or "free")
        print(
            f"[INFO] Video recording enabled: {self.video_path} "
            f"({self.width}x{self.height}@{self.fps:g}fps, camera={camera_desc})"
        )

    @staticmethod
    def _yaw_from_wxyz(quat: np.ndarray) -> float:
        qw, qx, qy, qz = quat
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def update_follow_camera(self, data: mujoco.MjData) -> mujoco.MjvCamera:
        root_pos = data.qpos[:3]
        yaw_deg = np.rad2deg(self._yaw_from_wxyz(data.qpos[3:7]))
        self.mjv_camera.lookat[:] = [root_pos[0], root_pos[1], root_pos[2] + self.follow_height]
        self.mjv_camera.distance = self.follow_distance
        self.mjv_camera.azimuth = yaw_deg + self.follow_yaw_offset
        self.mjv_camera.elevation = self.follow_elevation
        return self.mjv_camera

    def maybe_record(self, data: mujoco.MjData) -> None:
        if self.closed or data.time + 1e-9 < self.next_frame_time:
            return

        if self.follow_camera:
            self.renderer.update_scene(data, camera=self.update_follow_camera(data))
        elif self.camera is None:
            self.renderer.update_scene(data)
        else:
            self.renderer.update_scene(data, camera=self.camera)
        frame = self.renderer.render()
        self.writer.append_data(frame)
        self.frame_count += 1

        while self.next_frame_time <= data.time + 1e-9:
            self.next_frame_time += self.frame_interval

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.writer.close()
        self.renderer.close()
        print(f"[INFO] Video saved: {self.video_path} ({self.frame_count} frames)")


class G1DwaqMujocoRunner:
    """
    G1 DWAQ Sim2Sim 运行器 (带步态相位版本)
    
    加载 ActorCritic_DWAQ 策略和 MuJoCo 模型，运行实时仿真控制。
    步态相位用于实现稳定的两足交替行走。
    """

    def __init__(
        self,
        cfg: G1DwaqSim2SimCfg,
        checkpoint_path: str,
        model_path: str,
        record_video: bool = False,
        video_path: str | None = None,
        video_fps: float = 50.0,
        video_width: int = 640,
        video_height: int = 480,
        video_camera: str | None = None,
        video_follow_distance: float = 3.0,
        video_follow_height: float = 0.75,
        video_follow_yaw_offset: float = -45.0,
        video_follow_elevation: float = -28.0,
    ):
        self.cfg = cfg
        self.device = torch.device("cpu")  # MuJoCo sim2sim 用 CPU
        self.stop_requested = False
        self.record_video = bool(record_video)
        self.video_path = video_path
        self.video_fps = float(video_fps)
        self.video_width = int(video_width)
        self.video_height = int(video_height)
        self.video_camera = video_camera
        self.video_follow_distance = float(video_follow_distance)
        self.video_follow_height = float(video_follow_height)
        self.video_follow_yaw_offset = float(video_follow_yaw_offset)
        self.video_follow_elevation = float(video_follow_elevation)
        
        # 加载 MuJoCo 模型
        print(f"[INFO] 加载 MuJoCo 模型: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        
        # 加载策略
        print(f"[INFO] 加载 checkpoint: {checkpoint_path}")
        self.load_policy(checkpoint_path)
        
        # 初始化变量
        self.init_variables()
        self.build_joint_mappings()
        self.set_initial_pose()
        self.limit_monitor = self.create_limit_monitor(checkpoint_path)
        self.video_recorder = self.create_video_recorder(checkpoint_path) if self.record_video else None
        
        print(f"[INFO] 控制频率: {1.0 / (cfg.sim.dt * cfg.sim.decimation):.1f} Hz")
        print(f"[INFO] 观测维度: {cfg.sim.num_obs_per_step}")
        print(f"[INFO] 历史长度: {cfg.sim.dwaq_obs_history_length}")
        print(f"[INFO] Encoder 输入: {cfg.sim.num_obs_per_step * cfg.sim.dwaq_obs_history_length}")

    def create_limit_monitor(self, checkpoint_path: str) -> JointLimitMonitor:
        joint_names = getattr(self, 'mujoco_dof_names', MUJOCO_DOF_NAMES)
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        log_root = os.path.join(checkpoint_dir, 'sim2sim_limit_monitor')
        return JointLimitMonitor(self.model, joint_names, log_root)

    def create_video_recorder(self, checkpoint_path: str) -> MujocoVideoRecorder:
        if self.video_path:
            video_path = self.video_path
        else:
            checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            log_root = os.path.join(checkpoint_dir, 'sim2sim_videos')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_path = os.path.join(log_root, f'sim2sim_{timestamp}.mp4')
        return MujocoVideoRecorder(
            self.model,
            video_path,
            fps=self.video_fps,
            width=self.video_width,
            height=self.video_height,
            camera=self.video_camera,
            follow_distance=self.video_follow_distance,
            follow_height=self.video_follow_height,
            follow_yaw_offset=self.video_follow_yaw_offset,
            follow_elevation=self.video_follow_elevation,
        )

    def load_policy(self, checkpoint_path: str) -> None:
        """加载 ActorCritic_DWAQ 策略"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 构建网络参数
        num_obs = self.cfg.sim.num_obs_per_step
        obs_history_len = self.cfg.sim.dwaq_obs_history_length
        cenet_in_dim = num_obs * obs_history_len  # 96 * 5 = 480
        cenet_out_dim = self.cfg.sim.cenet_out_dim  # 19
        
        # Actor 输入 = 当前观测 + encoder 输出
        num_actor_obs = num_obs + cenet_out_dim  # 96 + 19 = 115
        
        policy_params = {
            'num_actor_obs': num_actor_obs,
            'num_critic_obs': 200,  # 不重要，推理时不用
            'num_actions': self.cfg.sim.num_actions,
            'cenet_in_dim': cenet_in_dim,
            'cenet_out_dim': cenet_out_dim,
            'obs_dim': num_obs,
            'activation': 'elu',
        }
        
        print(f"[INFO] 策略参数:")
        print(f"  - num_obs: {num_obs}")
        print(f"  - obs_history_len: {obs_history_len}")
        print(f"  - cenet_in_dim: {cenet_in_dim}")
        print(f"  - cenet_out_dim: {cenet_out_dim}")
        print(f"  - num_actor_obs: {num_actor_obs}")
        
        # 创建策略网络
        self.policy = ActorCritic_DWAQ(**policy_params)
        
        # 加载权重
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 只加载需要的权重 (actor + encoder)
        needed_keys = ['actor', 'encoder', 'encode_mean_latent', 'encode_mean_vel',
                       'encode_logvar_latent', 'encode_logvar_vel', 'decoder']
        filtered_state_dict = {}
        for key, value in model_state_dict.items():
            for needed in needed_keys:
                if key.startswith(needed):
                    filtered_state_dict[key] = value
                    break
        
        # 加载权重
        missing, unexpected = self.policy.load_state_dict(filtered_state_dict, strict=False)
        if missing:
            print(f"[WARNING] 缺少的权重: {missing[:5]}...")
        if unexpected:
            print(f"[WARNING] 多余的权重: {unexpected[:5]}...")
        
        self.policy.eval()
        print(f"[INFO] 策略加载完成")
        
        # 加载 normalizer (如果有)
        self.obs_normalizer = None
        if 'obs_norm_mean' in checkpoint and 'obs_norm_var' in checkpoint:
            self.obs_norm_mean = checkpoint['obs_norm_mean'].cpu().numpy()
            self.obs_norm_var = checkpoint['obs_norm_var'].cpu().numpy()
            print(f"[INFO] 加载观测归一化: mean shape={self.obs_norm_mean.shape}")
        else:
            self.obs_norm_mean = None
            self.obs_norm_var = None
            print(f"[INFO] 未找到观测归一化参数")

    def init_variables(self) -> None:
        """初始化仿真变量"""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_actions = self.cfg.sim.num_actions
        self.mujoco_dof_names = MUJOCO_DOF_NAMES
        self.lab_dof_names = LAB_DOF_NAMES
        
        # 关节状态
        self.dof_pos = np.zeros(self.num_actions)
        self.dof_vel = np.zeros(self.num_actions)
        
        # 动作 (Isaac Lab 顺序)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        
        # 默认关节位置 (MuJoCo 顺序) - 与 G1_CFG init_state.joint_pos 一致
        self.default_dof_pos = np.array([
            -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,    # 左腿
            -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,    # 右腿
            0.0, 0.0, 0.0,                        # 腰部
            0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0, # 左臂
            0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0 # 右臂
        ], dtype=np.float32)
        
        # PD 增益 (MuJoCo 顺序)
        self.kps = np.array([
            200, 150, 150, 200, 20, 20,    # 左腿
            200, 150, 150, 200, 20, 20,    # 右腿
            200, 200, 200,                  # 腰部
            100, 100, 50, 50, 40, 40, 40,  # 左臂
            100, 100, 50, 50, 40, 40, 40   # 右臂
        ], dtype=np.float32)
        
        self.kds = np.array([
            5, 5, 5, 5, 2, 2,              # 左腿
            5, 5, 5, 5, 2, 2,              # 右腿
            5, 5, 5,                        # 腰部
            2, 2, 2, 2, 2, 2, 2,           # 左臂
            2, 2, 2, 2, 2, 2, 2            # 右臂
        ], dtype=np.float32)
        
        self.episode_length_buf = 0
        
        # 速度命令
        self.command_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # 步态相位时间追踪
        self.gait_phase_time = 0.0  # 从仿真时间开始累积
        
        # DWAQ 观测历史缓冲 (5 帧 × 100 维，含步态相位)
        self.obs_history = np.zeros(
            (self.cfg.sim.dwaq_obs_history_length, self.cfg.sim.num_obs_per_step),
            dtype=np.float32
        )

    def build_joint_mappings(self) -> None:
        """建立关节映射索引"""
        # MuJoCo -> Isaac Lab 索引映射
        mujoco_indices = {name: idx for idx, name in enumerate(MUJOCO_DOF_NAMES)}
        self.mujoco_to_isaac_idx = [mujoco_indices[name] for name in LAB_DOF_NAMES]
        
        # Isaac Lab -> MuJoCo 索引映射
        lab_indices = {name: idx for idx, name in enumerate(LAB_DOF_NAMES)}
        self.isaac_to_mujoco_idx = [lab_indices[name] for name in MUJOCO_DOF_NAMES]
        
        # 默认关节位置转换为 Isaac Lab 顺序
        self.default_dof_pos_isaac = self.default_dof_pos[self.mujoco_to_isaac_idx]
        
        print(f"[INFO] 关节映射建立完成")

    def set_initial_pose(self) -> None:
        """设置初始姿态"""
        # 基座位置
        self.data.qpos[0:3] = [0.0, 0.0, self.cfg.robot.init_height]
        self.data.qpos[3:7] = [1, 0, 0, 0]  # (w, x, y, z)
        
        # 关节位置
        self.data.qpos[7:7 + self.num_actions] = self.default_dof_pos.copy()
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        print(f"[INFO] 初始高度: {self.data.qpos[2]:.3f}m")

    def get_gravity_orientation(self, quat: np.ndarray) -> np.ndarray:
        """计算投影重力向量
        
        Args:
            quat: MuJoCo 四元数 (w, x, y, z)
        
        Returns:
            投影重力向量 (3,)
        """
        qw, qx, qy, qz = quat
        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation

    def mj29_to_lab29(self, array_mj: np.ndarray) -> np.ndarray:
        """将 MuJoCo 顺序数组转换为 Isaac Lab 顺序"""
        return array_mj[self.mujoco_to_isaac_idx]

    def lab29_to_mj29(self, array_lab: np.ndarray) -> np.ndarray:
        """将 Isaac Lab 顺序数组转换为 MuJoCo 顺序"""
        return array_lab[self.isaac_to_mujoco_idx]

    def get_current_obs(self) -> np.ndarray:
        """计算当前单帧观测向量 (100 维，含步态相位)"""
        # 读取关节状态 (MuJoCo 顺序)
        dof_pos_mj = self.data.qpos[7:7 + self.num_actions].copy()
        dof_vel_mj = self.data.qvel[6:6 + self.num_actions].copy()
        
        # 基座状态
        ang_vel_body = self.data.qvel[3:6].copy()
        
        # 投影重力
        quat = self.data.qpos[3:7].copy()  # MuJoCo: (w, x, y, z)
        projected_gravity = self.get_gravity_orientation(quat)
        
        # 转换为 Isaac Lab 顺序
        joint_pos_isaac = self.mj29_to_lab29(dof_pos_mj - self.default_dof_pos)
        joint_vel_isaac = self.mj29_to_lab29(dof_vel_mj)
        
        # 计算步态相位 (4 维)
        gait_phase = self.compute_gait_phase()
        
        # 构建观测 (100 维) - 带步态相位
        obs = np.concatenate([
            ang_vel_body,                     # 3: 角速度 (body frame)
            projected_gravity,                # 3: 投影重力 (body frame)
            self.command_vel,                 # 3: 速度命令
            joint_pos_isaac,                  # 29: 关节位置偏差
            joint_vel_isaac,                  # 29: 关节速度
            np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),  # 29: 上一步动作
            gait_phase,                       # 4: 步态相位 [sin_L, cos_L, sin_R, cos_R]
        ], axis=0).astype(np.float32)
        
        return np.clip(obs, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)
    
    def compute_gait_phase(self) -> np.ndarray:
        """计算步态相位 (4 维: sin_left, cos_left, sin_right, cos_right)
        
        左右腿相位差 offset=0.5，实现交替行走。
        """
        period = self.cfg.gait_phase.period  # 0.8s
        offset = self.cfg.gait_phase.offset  # 0.5
        
        # 左腿相位
        phase_left = (self.gait_phase_time % period) / period  # [0, 1)
        
        # 右腿相位 (偏移 50%)
        phase_right = ((self.gait_phase_time / period) + offset) % 1.0
        
        # 转换为 sin/cos
        sin_left = np.sin(2 * np.pi * phase_left)
        cos_left = np.cos(2 * np.pi * phase_left)
        sin_right = np.sin(2 * np.pi * phase_right)
        cos_right = np.cos(2 * np.pi * phase_right)
        
        return np.array([sin_left, cos_left, sin_right, cos_right], dtype=np.float32)

    def update_obs_history(self, current_obs: np.ndarray) -> None:
        """更新观测历史缓冲 (FIFO)"""
        # 将历史向前滚动，新观测放在最后
        self.obs_history = np.roll(self.obs_history, shift=-1, axis=0)
        self.obs_history[-1] = current_obs.copy()

    def get_flattened_obs_history(self) -> np.ndarray:
        """获取扁平化的观测历史 (5 * 100 = 500 维)"""
        return self.obs_history.flatten()

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """归一化观测 (如果有 normalizer)"""
        if self.obs_norm_mean is not None and self.obs_norm_var is not None:
            return (obs - self.obs_norm_mean) / np.sqrt(self.obs_norm_var + 1e-8)
        return obs

    def position_control(self) -> np.ndarray:
        """计算目标关节位置 (MuJoCo 顺序)"""
        actions_scaled = self.action * self.cfg.sim.action_scale
        return self.lab29_to_mj29(actions_scaled) + self.default_dof_pos

    def pd_control(self, target_q: np.ndarray) -> np.ndarray:
        """PD 控制器计算力矩"""
        q = self.data.qpos[7:7 + self.num_actions]
        dq = self.data.qvel[6:6 + self.num_actions]
        return (target_q - q) * self.kps + (0 - dq) * self.kds

    def run(self) -> None:
        """运行仿真循环"""
        self.setup_keyboard_listener()
        self.listener.start()
        
        # 创建 viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # 初始化观测历史 (填充初始观测)
        initial_obs = self.get_current_obs()
        for i in range(self.cfg.sim.dwaq_obs_history_length):
            self.obs_history[i] = initial_obs.copy()
        
        print("\n[INFO] 键盘控制: 8/2=前后, 4/6=左右, 7/9=转向, 5=停止")
        if self.video_recorder is not None:
            print("[INFO] Recording video; press q to stop and save MP4")
        print("[INFO] 按 q 或 Ctrl+C 退出\n")
        
        debug_counter = 0
        
        try:
            while (
                self.viewer.is_running()
                and not self.stop_requested
                and self.data.time < self.cfg.sim.sim_duration
            ):
                # 获取当前观测
                current_obs = self.get_current_obs()
                
                # 归一化当前观测
                current_obs_normalized = self.normalize_obs(current_obs)
                
                # 更新观测历史 (使用归一化后的观测)
                self.update_obs_history(current_obs_normalized)
                
                # 获取扁平化的观测历史
                obs_history_flat = self.get_flattened_obs_history()
                
                # 转换为 tensor
                obs_tensor = torch.tensor(current_obs_normalized, dtype=torch.float32).unsqueeze(0)
                obs_history_tensor = torch.tensor(obs_history_flat, dtype=torch.float32).unsqueeze(0)
                
                # 执行策略
                with torch.no_grad():
                    action_tensor = self.policy.act_inference(obs_tensor, obs_history_tensor)
                
                self.action[:] = action_tensor.squeeze(0).numpy()
                self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
                
                # 调试输出
                debug_counter += 1
                if debug_counter <= 3:
                    print(f"\n[DEBUG] Step {debug_counter}")
                    print(f"  current_obs shape: {current_obs.shape}")
                    print(f"  obs_history shape: {self.obs_history.shape}")
                    print(f"  obs_history_flat shape: {obs_history_flat.shape}")
                    print(f"  command: {self.command_vel}")
                    print(f"  action (first 6): {self.action[:6]}")
                
                # 执行 decimation 步
                for _ in range(self.cfg.sim.decimation):
                    step_start = time.time()
                    
                    # 计算目标位置并执行 PD 控制
                    target_pos = self.position_control()
                    tau = self.pd_control(target_pos)
                    self.limit_monitor.record(
                        self.data.time,
                        self.data.qpos[7:7 + self.num_actions],
                        target_pos,
                        tau,
                    )
                    self.data.ctrl[:self.num_actions] = tau
                    
                    # 物理步进
                    mujoco.mj_step(self.model, self.data)
                    if self.video_recorder is not None:
                        self.video_recorder.maybe_record(self.data)
                    self.viewer.sync()
                    
                    # 时间控制
                    elapsed = time.time() - step_start
                    if self.cfg.sim.dt - elapsed > 0:
                        time.sleep(self.cfg.sim.dt - elapsed)
                
                # 更新步态相位时间 (每个控制周期，与训练一致)
                control_dt = self.cfg.sim.dt * self.cfg.sim.decimation
                self.gait_phase_time += control_dt
                
                self.episode_length_buf += 1
                
                # 定期打印状态
                if self.episode_length_buf % 100 == 0:
                    print(f"[INFO] t={self.data.time:.1f}s, cmd=[{self.command_vel[0]:.2f}, {self.command_vel[1]:.2f}, {self.command_vel[2]:.2f}], h={self.data.qpos[2]:.3f}m")
                    
        except KeyboardInterrupt:
            print("\n[INFO] 用户中断")
        finally:
            if getattr(self, 'video_recorder', None) is not None:
                self.video_recorder.close()
            if hasattr(self, 'limit_monitor'):
                self.limit_monitor.save()
            if hasattr(self, 'listener'):
                self.listener.stop()
            if hasattr(self, 'viewer'):
                self.viewer.close()
            print("[INFO] 仿真结束")

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """调整速度命令"""
        limits = [1.0, 0.5, 1.57]  # vx, vy, yaw_rate 限制
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -limits[idx], limits[idx])

    def setup_keyboard_listener(self) -> None:
        """设置键盘监听器"""
        # 小键盘 virtual key codes (Linux/Windows)
        # NumPad: 0-9 对应 vk 96-105 (Windows) 或特定的 vk 值
        NUMPAD_VK = {
            65437: "5",  # KP_5 (Linux)
            65429: "7",  # KP_7 (Linux) 
            65431: "9",  # KP_9 (Linux)
            65433: "4",  # KP_4 (Linux)
            65435: "6",  # KP_6 (Linux)
            65438: "2",  # KP_2 (Linux)
            65436: "8",  # KP_8 (Linux)
            65430: "1",  # KP_1 (Linux)
            65432: "3",  # KP_3 (Linux)
            96: "0", 97: "1", 98: "2", 99: "3", 100: "4",  # Windows
            101: "5", 102: "6", 103: "7", 104: "8", 105: "9",
        }
        
        def handle_key(char):
            """处理按键字符"""
            if char == "8": self.adjust_command_vel(0, 0.2)    # 前进
            elif char == "2": self.adjust_command_vel(0, -0.2)  # 后退
            elif char == "4": self.adjust_command_vel(1, 0.2)   # 左移
            elif char == "6": self.adjust_command_vel(1, -0.2)  # 右移
            elif char == "7": self.adjust_command_vel(2, 0.3)   # 左转
            elif char == "9": self.adjust_command_vel(2, -0.3)  # 右转
            elif char == "5": 
                self.command_vel[0] = 0.0
                self.command_vel[1] = 0.0
                self.command_vel[2] = 0.0
                print("[INFO] 命令已清零")
        
        def on_press(key):
            # 主键盘 q: stop simulation and finalize video.
            try:
                if key.char is not None:
                    char = key.char.lower()
                    if char == "q":
                        self.stop_requested = True
                        print("[INFO] q pressed: stopping sim2sim and saving outputs...")
                        return False
                    if char in "123456789":
                        handle_key(char)
                        return
            except AttributeError:
                pass
            
            # 尝试小键盘数字键 (通过 vk 码)
            try:
                vk = key.vk
                if vk in NUMPAD_VK:
                    handle_key(NUMPAD_VK[vk])
            except AttributeError:
                pass
        
        self.listener = keyboard.Listener(on_press=on_press)


def get_available_scenes(mjcf_dir: str) -> dict:
    """获取可用的场景文件"""
    scenes = {}
    
    if os.path.isdir(mjcf_dir):
        for f in os.listdir(mjcf_dir):
            if f.endswith('.xml'):
                if f in ['scene.xml', 'stairs.xml', 'flat_scene.xml', 'rough_scene.xml', 'slope_scene.xml']:
                    name = f.replace('.xml', '').replace('_scene', '')
                    scenes[name] = os.path.join(mjcf_dir, f)
                elif f.endswith('_scene.xml'):
                    name = f.replace('_scene.xml', '')
                    scenes[name] = os.path.join(mjcf_dir, f)
    
    return scenes


def find_latest_checkpoint(logs_dir: str, task_name: str) -> str:
    """查找最新的 checkpoint 文件
    
    Returns:
        最新的 model_xxxx.pt 路径，如果找不到则返回 None
    """
    task_dir = os.path.join(logs_dir, task_name)
    if not os.path.isdir(task_dir):
        return None
    
    # 获取所有日期目录并排序
    date_dirs = []
    for d in os.listdir(task_dir):
        full_path = os.path.join(task_dir, d)
        if os.path.isdir(full_path):
            date_dirs.append((d, full_path))
    
    if not date_dirs:
        return None
    
    # 按名称排序，最新的在最后
    date_dirs.sort(key=lambda x: x[0])
    latest_dir = date_dirs[-1][1]
    
    # 查找最新的 model_xxxx.pt
    model_files = []
    for f in os.listdir(latest_dir):
        if f.startswith('model_') and f.endswith('.pt'):
            # 提取迭代数
            try:
                iter_num = int(f.replace('model_', '').replace('.pt', ''))
                model_files.append((iter_num, os.path.join(latest_dir, f)))
            except ValueError:
                pass
    
    if model_files:
        model_files.sort(key=lambda x: x[0])
        return model_files[-1][1]
    
    return None


def main():
    LEGGED_LAB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    MJCF_DIR = os.path.join(LEGGED_LAB_ROOT, "legged_lab/assets/unitree/g1/mjcf")
    LOGS_DIR = os.path.join(LEGGED_LAB_ROOT, "logs")
    
    # 尝试查找最新的 g1_dwaq checkpoint
    # default_checkpoint = find_latest_checkpoint(LOGS_DIR, "g1_dwaq")
    # if default_checkpoint is None:
    #     # 回退到硬编码路径
    #     default_checkpoint = os.path.join(LOGS_DIR, "g1_dwaq/2026-01-15_11-21-04/model_10000.pt")

    default_checkpoint = os.path.join(LOGS_DIR, "g1_dwaq/2026-01-16_00-46-00/model_9999.pt")

    default_model = os.path.join(MJCF_DIR, "g1_29dof_rev_1_0_daf.xml")
    
    # 获取可用场景
    available_scenes = get_available_scenes(MJCF_DIR)
    scene_names = list(available_scenes.keys())
    
    parser = argparse.ArgumentParser(
        description="G1 DWAQ Sim2Sim - 盲走策略 (无步态版本)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, 
                        help="模型 checkpoint 路径 (model_xxxx.pt)")
    parser.add_argument("--model", type=str, default=default_model,
                        help="MuJoCo XML 模型路径")
    parser.add_argument("--scene", type=str, default=None,
                        choices=scene_names if scene_names else None,
                        help=f"场景名称，可用场景: {', '.join(scene_names) if scene_names else '无'}")
    parser.add_argument("--scene-file", type=str, default=None,
                        help="直接指定场景文件路径 (优先级高于 --scene)")
    parser.add_argument("--duration", type=float, default=100.0, help="仿真时长 (秒)")
    parser.add_argument("--record-video", action="store_true", help="录制 MuJoCo viewer MP4，按 q 结束并保存")
    parser.add_argument("--video-path", type=str, default=None, help="MP4 保存路径；默认保存到 checkpoint 目录下 sim2sim_videos")
    parser.add_argument("--video-fps", type=float, default=50.0, help="录制帧率")
    parser.add_argument("--video-width", type=int, default=640, help="录制宽度")
    parser.add_argument("--video-height", type=int, default=480, help="录制高度")
    parser.add_argument("--video-camera", type=str, default=None, help="MuJoCo camera 名称；默认 follow_side_rear，可传 free 禁用跟随")
    parser.add_argument("--video-follow-distance", type=float, default=3.0, help="跟随视角距离")
    parser.add_argument("--video-follow-height", type=float, default=0.75, help="跟随视角看向 root 上方高度")
    parser.add_argument("--video-follow-yaw-offset", type=float, default=-45.0, help="相对机器人 yaw 的侧后方角度偏移，默认在机器人背后")
    parser.add_argument("--video-follow-elevation", type=float, default=-28.0, help="跟随视角俯仰角，默认更偏俯视")
    parser.add_argument("--list-scenes", action="store_true", help="列出所有可用场景")
    args = parser.parse_args()
    
    # 列出可用场景
    if args.list_scenes:
        print("\n可用场景:")
        print("-" * 40)
        for name, path in available_scenes.items():
            print(f"  {name:15} -> {os.path.basename(path)}")
        print("-" * 40)
        print(f"场景文件目录: {MJCF_DIR}")
        sys.exit(0)
    
    # 检查 checkpoint 文件
    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint 文件不存在: {args.checkpoint}")
        print(f"[INFO] 请指定正确的 checkpoint 路径，例如:")
        print(f"       python {sys.argv[0]} --checkpoint logs/g1_dwaq/2026-xx-xx/model_10000.pt")
        sys.exit(1)
    
    # 确定要加载的模型/场景文件
    if args.scene_file:
        model_path = args.scene_file
        if not os.path.isfile(model_path):
            print(f"[ERROR] 场景文件不存在: {model_path}")
            sys.exit(1)
    elif args.scene:
        if args.scene not in available_scenes:
            print(f"[ERROR] 未知场景: {args.scene}")
            sys.exit(1)
        model_path = available_scenes[args.scene]
    else:
        model_path = args.model
        if not os.path.isfile(model_path):
            print(f"[ERROR] MuJoCo 模型不存在: {model_path}")
            sys.exit(1)
    
    print("=" * 60)
    print("G1 DWAQ Sim2Sim")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"MuJoCo 模型/场景: {model_path}")
    if args.scene:
        print(f"场景: {args.scene}")
    print(f"模式: 盲走 (blind walking with VAE)")
    print(f"历史: 5 帧")
    print("=" * 60)
    
    cfg = G1DwaqSim2SimCfg()
    cfg.sim.sim_duration = args.duration
    
    runner = G1DwaqMujocoRunner(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        model_path=model_path,
    )
    runner.run()


if __name__ == "__main__":
    main()
