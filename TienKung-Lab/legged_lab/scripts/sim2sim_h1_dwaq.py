"""Original H1 DWAQ Sim2Sim runner.

This reuses the proven G1 DWAQ runner implementation and overrides only the
robot-specific mappings/configuration for original H1 (19 DoF).
"""

import argparse
import os
import sys

import numpy as np

try:
    from . import sim2sim_g1_dwaq as g1_sim2sim
except ImportError:
    import sim2sim_g1_dwaq as g1_sim2sim


# MuJoCo joint order in h1_description/mjcf/h1.xml (19 DoF)
MUJOCO_DOF_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

# Isaac-Lab action order used by policy training (19 DoF)
LAB_DOF_NAMES = [
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "torso_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_joint",
    "right_ankle_joint",
    "left_elbow_joint",
    "right_elbow_joint",
]


class H1DwaqSim2SimCfg:
    class sim:
        sim_duration = 100.0
        num_actions = 19
        # 3 + 3 + 3 + 19 + 19 + 19 + 4
        num_obs_per_step = 70
        dwaq_obs_history_length = 5
        dt = 0.005
        decimation = 4
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25
        cenet_out_dim = 19

    class robot:
        init_height = 1.05

    class gait_phase:
        enable = True
        period = 0.8
        offset = 0.5


class H1DwaqMujocoRunner(g1_sim2sim.G1DwaqMujocoRunner):
    """H1-specialized runner with 19-DoF mapping and observation layout."""

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_actions = self.cfg.sim.num_actions

        self.dof_pos = np.zeros(self.num_actions)
        self.dof_vel = np.zeros(self.num_actions)
        self.action = np.zeros(self.num_actions, dtype=np.float32)

        # MuJoCo order, aligned with H1_CFG init_state defaults.
        self.default_dof_pos = np.array(
            [
                0.0,
                0.0,
                -0.28,
                0.79,
                -0.52,
                0.0,
                0.0,
                -0.28,
                0.79,
                -0.52,
                0.0,
                0.20,
                0.0,
                0.0,
                0.32,
                0.20,
                0.0,
                0.0,
                0.32,
            ],
            dtype=np.float32,
        )

        # MuJoCo-order PD gains from H1 actuator configuration.
        self.kps = np.array(
            [
                200.0,
                200.0,
                200.0,
                300.0,
                40.0,
                200.0,
                200.0,
                200.0,
                300.0,
                40.0,
                300.0,
                100.0,
                50.0,
                50.0,
                50.0,
                100.0,
                50.0,
                50.0,
                50.0,
            ],
            dtype=np.float32,
        )
        self.kds = np.array(
            [
                5.0,
                5.0,
                5.0,
                6.0,
                2.0,
                5.0,
                5.0,
                5.0,
                6.0,
                2.0,
                6.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],
            dtype=np.float32,
        )

        self.episode_length_buf = 0
        self.command_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.gait_phase_time = 0.0
        self.obs_history = np.zeros(
            (self.cfg.sim.dwaq_obs_history_length, self.cfg.sim.num_obs_per_step),
            dtype=np.float32,
        )

    def build_joint_mappings(self) -> None:
        mujoco_indices = {name: idx for idx, name in enumerate(MUJOCO_DOF_NAMES)}
        self.mujoco_to_isaac_idx = [mujoco_indices[name] for name in LAB_DOF_NAMES]

        lab_indices = {name: idx for idx, name in enumerate(LAB_DOF_NAMES)}
        self.isaac_to_mujoco_idx = [lab_indices[name] for name in MUJOCO_DOF_NAMES]
        self.default_dof_pos_isaac = self.default_dof_pos[self.mujoco_to_isaac_idx]
        print("[INFO] Joint mapping initialized for H1")

    def mj29_to_lab29(self, array_mj: np.ndarray) -> np.ndarray:
        return array_mj[self.mujoco_to_isaac_idx]

    def lab29_to_mj29(self, array_lab: np.ndarray) -> np.ndarray:
        return array_lab[self.isaac_to_mujoco_idx]

    def compute_gait_phase(self) -> np.ndarray:
        """Return [sin_left, sin_right, cos_left, cos_right] to match Isaac."""
        period = self.cfg.gait_phase.period
        offset = self.cfg.gait_phase.offset

        phase_left = (self.gait_phase_time % period) / period
        phase_right = ((self.gait_phase_time / period) + offset) % 1.0

        sin_left = np.sin(2 * np.pi * phase_left)
        sin_right = np.sin(2 * np.pi * phase_right)
        cos_left = np.cos(2 * np.pi * phase_left)
        cos_right = np.cos(2 * np.pi * phase_right)
        return np.array([sin_left, sin_right, cos_left, cos_right], dtype=np.float32)

    def get_current_obs(self) -> np.ndarray:
        # MuJoCo-order states
        dof_pos_mj = self.data.qpos[7 : 7 + self.num_actions].copy()
        dof_vel_mj = self.data.qvel[6 : 6 + self.num_actions].copy()
        ang_vel_body = self.data.qvel[3:6].copy()
        quat = self.data.qpos[3:7].copy()
        projected_gravity = self.get_gravity_orientation(quat)

        # Convert to Isaac action order
        joint_pos_isaac = self.mj29_to_lab29(dof_pos_mj - self.default_dof_pos)
        joint_vel_isaac = self.mj29_to_lab29(dof_vel_mj)
        prev_action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)

        obs_terms = [
            ang_vel_body,
            projected_gravity,
            self.command_vel,
            joint_pos_isaac,
            joint_vel_isaac,
            prev_action,
        ]

        if self.cfg.gait_phase.enable:
            obs_terms.append(self.compute_gait_phase())

        obs = np.concatenate(obs_terms, axis=0).astype(np.float32)
        return np.clip(obs, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)


def main():
    legged_lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    mjcf_dir = os.path.join(legged_lab_root, "legged_lab/assets/h1_description/mjcf")
    logs_dir = os.path.join(legged_lab_root, "logs")

    default_checkpoint = g1_sim2sim.find_latest_checkpoint(logs_dir, "h1_dwaq")
    if default_checkpoint is None:
        default_checkpoint = os.path.join(logs_dir, "h1_dwaq/model_10000.pt")

    default_model = os.path.join(mjcf_dir, "h1.xml")

    available_scenes = g1_sim2sim.get_available_scenes(mjcf_dir)
    if os.path.isfile(default_model):
        available_scenes["h1"] = default_model
    scene_names = list(available_scenes.keys())

    parser = argparse.ArgumentParser(
        description="Original H1 DWAQ Sim2Sim",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, help="checkpoint path")
    parser.add_argument("--model", type=str, default=default_model, help="MuJoCo XML model path")
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        choices=scene_names if scene_names else None,
        help=f"scene name: {', '.join(scene_names) if scene_names else 'none'}",
    )
    parser.add_argument("--scene-file", type=str, default=None, help="explicit scene file path")
    parser.add_argument("--duration", type=float, default=100.0, help="simulation duration in seconds")
    parser.add_argument("--list-scenes", action="store_true", help="list available scenes and exit")
    args = parser.parse_args()

    if args.list_scenes:
        print("\nAvailable scenes:")
        print("-" * 40)
        for name, path in available_scenes.items():
            print(f"  {name:15} -> {os.path.basename(path)}")
        print("-" * 40)
        print(f"Scene directory: {mjcf_dir}")
        sys.exit(0)

    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint file not found: {args.checkpoint}")
        print(f"[INFO] Example: python {sys.argv[0]} --checkpoint logs/h1_dwaq/<run>/model_10000.pt")
        sys.exit(1)

    if args.scene_file:
        model_path = args.scene_file
        if not os.path.isfile(model_path):
            print(f"[ERROR] Scene file not found: {model_path}")
            sys.exit(1)
    elif args.scene:
        if args.scene not in available_scenes:
            print(f"[ERROR] Unknown scene: {args.scene}")
            sys.exit(1)
        model_path = available_scenes[args.scene]
    else:
        model_path = args.model
        if not os.path.isfile(model_path):
            print(f"[ERROR] MuJoCo model file not found: {model_path}")
            sys.exit(1)

    print("=" * 60)
    print("H1 DWAQ Sim2Sim")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"MuJoCo model/scene: {model_path}")
    if args.scene:
        print(f"Scene: {args.scene}")
    print("Mode: blind walking with DWAQ context encoder")
    print("History: 5 frames")
    print("=" * 60)

    cfg = H1DwaqSim2SimCfg()
    cfg.sim.sim_duration = args.duration

    runner = H1DwaqMujocoRunner(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        model_path=model_path,
    )
    runner.run()


if __name__ == "__main__":
    main()
