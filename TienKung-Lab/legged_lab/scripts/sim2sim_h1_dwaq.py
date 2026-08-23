"""Original H1 DWAQ Sim2Sim runner.

This reuses the proven G1 DWAQ runner implementation and overrides only the
robot-specific mappings/configuration for original H1 (19 DoF).
"""

import argparse
import os
import sys

import mujoco
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
        standing_command_threshold = 0.1


class H1DwaqMujocoRunner(g1_sim2sim.G1DwaqMujocoRunner):
    """H1-specialized runner with 19-DoF mapping and observation layout."""

    def init_variables(self) -> None:
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_actions = self.cfg.sim.num_actions
        self.mujoco_dof_names = MUJOCO_DOF_NAMES
        self.lab_dof_names = LAB_DOF_NAMES

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
        self.auto_reset_on_flat_return = False
        self.auto_reset_climbed_height = 1.15
        self.auto_reset_flat_x_min = 1.14
        self.auto_reset_flat_x_max = 7.37
        self.auto_reset_armed = False
        self.auto_reset_count = 0

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

        command_magnitude = np.linalg.norm(self.command_vel[:2]) + abs(self.command_vel[2])
        if command_magnitude < self.cfg.gait_phase.standing_command_threshold:
            # Match training: freeze at a double-support phase for zero-speed
            # commands instead of asking the policy to alternate its feet.
            self.gait_phase_time = 0.0
            phase_left = 0.0
            phase_right = offset
        else:
            phase_left = (self.gait_phase_time % period) / period
            phase_right = ((self.gait_phase_time / period) + offset) % 1.0

        sin_left = np.sin(2 * np.pi * phase_left)
        sin_right = np.sin(2 * np.pi * phase_right)
        cos_left = np.cos(2 * np.pi * phase_left)
        cos_right = np.cos(2 * np.pi * phase_right)
        return np.array([sin_left, sin_right, cos_left, cos_right], dtype=np.float32)

    def maybe_auto_reset_after_flat_return(self) -> None:
        """Reset after a climbed robot returns to either flat end of the stair course."""
        if not self.auto_reset_on_flat_return:
            return

        root_x = float(self.data.qpos[0])
        root_height = float(self.data.qpos[2])
        inside_stair_course = self.auto_reset_flat_x_min < root_x < self.auto_reset_flat_x_max

        if not self.auto_reset_armed:
            if inside_stair_course and root_height >= self.auto_reset_climbed_height:
                self.auto_reset_armed = True
                print(
                    f"[INFO] Auto-reset armed at x={root_x:.3f}m, "
                    f"h={root_height:.3f}m, t={self.data.time:.2f}s"
                )
            return

        returned_to_flat = (
            root_x <= self.auto_reset_flat_x_min
            or root_x >= self.auto_reset_flat_x_max
        )
        if not returned_to_flat:
            return

        reset_time = float(self.data.time)
        self.auto_reset_count += 1
        print(
            f"[INFO] Auto-reset #{self.auto_reset_count}: returned to flat at "
            f"x={root_x:.3f}m, h={root_height:.3f}m, t={reset_time:.2f}s"
        )
        self.set_initial_pose()
        # Preserve total simulation time so repeated attempts cannot extend the
        # requested video duration indefinitely.
        self.data.time = reset_time
        self.action.fill(0.0)
        self.obs_history.fill(0.0)
        self.gait_phase_time = 0.0
        self.episode_length_buf = 0
        self.command_vx_switch_triggered = False
        self.command_vx_second_switch_triggered = False
        self.auto_reset_armed = False
        mujoco.mj_forward(self.model, self.data)

    def get_current_obs(self) -> np.ndarray:
        self.maybe_auto_reset_after_flat_return()

        command_schedule = getattr(self, "command_vx_schedule", ())
        schedule_index = getattr(self, "command_vx_schedule_index", 0)
        command_vx_limit = getattr(self, "command_vx_limit", 1.0)
        while (
            schedule_index < len(command_schedule)
            and self.data.time >= command_schedule[schedule_index][0]
        ):
            switch_time, scheduled_vx = command_schedule[schedule_index]
            previous_vx = float(self.command_vel[0])
            self.command_vel[0] = np.clip(
                scheduled_vx, -command_vx_limit, command_vx_limit
            )
            schedule_index += 1
            self.command_vx_schedule_index = schedule_index
            print(
                f"[INFO] Scheduled forward command switch at "
                f"x={self.data.qpos[0]:.3f}m, h={self.data.qpos[2]:.3f}m, "
                f"t={self.data.time:.2f}s (target {switch_time:.2f}s): "
                f"{previous_vx:.2f} -> {self.command_vel[0]:.2f} m/s"
            )

        switch_x = getattr(self, "command_vx_switch_x", None)
        switch_height = getattr(self, "command_vx_switch_height", None)
        switch_time = getattr(self, "command_vx_switch_time", None)
        switch_vx = getattr(self, "command_vx_after_switch", None)
        switch_triggered = getattr(self, "command_vx_switch_triggered", False)
        switch_trigger_reached = (
            (switch_x is not None and self.data.qpos[0] >= switch_x)
            or (switch_height is not None and self.data.qpos[2] >= switch_height)
            or (switch_time is not None and self.data.time >= switch_time)
        )
        if (
            (switch_x is not None or switch_height is not None or switch_time is not None)
            and switch_vx is not None
            and not switch_triggered
            and switch_trigger_reached
        ):
            previous_vx = float(self.command_vel[0])
            self.command_vel[0] = np.clip(
                switch_vx, -command_vx_limit, command_vx_limit
            )
            self.command_vx_switch_triggered = True
            print(
                f"[INFO] Forward command switched at x={self.data.qpos[0]:.3f}m, "
                f"h={self.data.qpos[2]:.3f}m, "
                f"t={self.data.time:.2f}s: {previous_vx:.2f} -> {self.command_vel[0]:.2f} m/s"
            )

        second_switch_height = getattr(self, "command_vx_second_switch_height", None)
        second_switch_vx = getattr(self, "command_vx_after_second_switch", None)
        second_switch_triggered = getattr(self, "command_vx_second_switch_triggered", False)
        if (
            self.command_vx_switch_triggered
            and second_switch_height is not None
            and second_switch_vx is not None
            and not second_switch_triggered
            and self.data.qpos[2] >= second_switch_height
        ):
            previous_vx = float(self.command_vel[0])
            self.command_vel[0] = np.clip(
                second_switch_vx, -command_vx_limit, command_vx_limit
            )
            self.command_vx_second_switch_triggered = True
            print(
                f"[INFO] Second forward command switch at x={self.data.qpos[0]:.3f}m, "
                f"h={self.data.qpos[2]:.3f}m, "
                f"t={self.data.time:.2f}s: {previous_vx:.2f} -> {self.command_vel[0]:.2f} m/s"
            )

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
    payload_model = os.path.join(mjcf_dir, "h1_payload_horizontal.xml")
    if os.path.isfile(payload_model):
        available_scenes["h1_payload"] = payload_model
    payload_stairs_model = os.path.join(mjcf_dir, "scene_payload_horizontal.xml")
    if os.path.isfile(payload_stairs_model):
        available_scenes["h1_payload_stairs"] = payload_stairs_model
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
    parser.add_argument(
        "--auto-reset-on-flat-return",
        action="store_true",
        help="after climbing, reset when the robot returns to either flat end of the stair course",
    )
    parser.add_argument(
        "--auto-reset-climbed-height",
        type=float,
        default=1.15,
        help="base height that arms automatic flat-return reset",
    )
    parser.add_argument(
        "--auto-reset-flat-x-min",
        type=float,
        default=1.14,
        help="x coordinate at or below which the robot is back on the approach flat",
    )
    parser.add_argument(
        "--auto-reset-flat-x-max",
        type=float,
        default=7.37,
        help="x coordinate at or above which the robot has reached the far flat",
    )
    parser.add_argument(
        "--command-vx",
        type=float,
        default=0.0,
        help="initial forward velocity command in m/s; 0.2 matches one press of key 8",
    )
    parser.add_argument(
        "--command-vx-switch-x",
        type=float,
        default=None,
        help="switch the forward command when the base reaches this world x position",
    )
    parser.add_argument(
        "--command-vx-after-switch",
        type=float,
        default=None,
        help="forward velocity command used after the configured position/height trigger",
    )
    parser.add_argument(
        "--command-vx-switch-height",
        type=float,
        default=None,
        help="switch the forward command when the base reaches this world height",
    )
    parser.add_argument(
        "--command-vx-switch-time",
        type=float,
        default=None,
        help="switch the forward command when simulation time reaches this value in seconds",
    )
    parser.add_argument(
        "--command-vx-second-switch-height",
        type=float,
        default=None,
        help="after the first switch, switch again when base height reaches this value",
    )
    parser.add_argument(
        "--command-vx-after-second-switch",
        type=float,
        default=None,
        help="forward velocity command used after the second height trigger",
    )
    parser.add_argument(
        "--command-vx-schedule",
        type=str,
        default=None,
        help='comma-separated time:velocity stages, for example "10:0.6,20:0.8"',
    )
    parser.add_argument(
        "--command-vx-limit",
        type=float,
        default=1.0,
        help="absolute sim-only forward command limit; use 1.2 for an out-of-training-range stress test",
    )
    parser.add_argument("--record-video", action="store_true", help="record MuJoCo MP4; press q to stop and save")
    parser.add_argument("--video-path", type=str, default=None, help="MP4 output path; default saves beside checkpoint")
    parser.add_argument("--video-fps", type=float, default=50.0, help="recording FPS")
    parser.add_argument("--video-width", type=int, default=640, help="recording width")
    parser.add_argument("--video-height", type=int, default=480, help="recording height")
    parser.add_argument("--video-camera", type=str, default=None, help="MuJoCo camera name; default follows from side-rear, pass free to disable")
    parser.add_argument("--video-follow-distance", type=float, default=3.0, help="follow camera distance")
    parser.add_argument("--video-follow-height", type=float, default=0.75, help="follow camera look-at height above root")
    parser.add_argument("--video-follow-yaw-offset", type=float, default=-45.0, help="side-rear yaw offset relative to robot heading; default is behind the robot")
    parser.add_argument("--video-follow-elevation", type=float, default=-28.0, help="follow camera elevation angle; default is more top-down")
    parser.add_argument("--list-scenes", action="store_true", help="list available scenes and exit")
    args = parser.parse_args()
    if args.command_vx_limit <= 0.0:
        parser.error("--command-vx-limit must be positive")

    command_vx_schedule = []
    if args.command_vx_schedule:
        try:
            for stage in args.command_vx_schedule.split(","):
                switch_time_text, velocity_text = stage.split(":", maxsplit=1)
                command_vx_schedule.append(
                    (float(switch_time_text), float(velocity_text))
                )
        except ValueError:
            parser.error(
                "--command-vx-schedule must contain comma-separated "
                "time:velocity pairs"
            )
        if any(stage[0] < 0.0 for stage in command_vx_schedule):
            parser.error("--command-vx-schedule times must be non-negative")
        if any(
            next_stage[0] <= current_stage[0]
            for current_stage, next_stage in zip(
                command_vx_schedule, command_vx_schedule[1:]
            )
        ):
            parser.error("--command-vx-schedule times must be strictly increasing")

    switch_trigger_count = sum(
        value is not None
        for value in (
            args.command_vx_switch_x,
            args.command_vx_switch_height,
            args.command_vx_switch_time,
        )
    )
    if switch_trigger_count > 1:
        parser.error(
            "provide only one of --command-vx-switch-x, "
            "--command-vx-switch-height, or --command-vx-switch-time"
        )
    if (switch_trigger_count == 0) != (args.command_vx_after_switch is None):
        parser.error("a command switch trigger and --command-vx-after-switch must be provided together")
    if (args.command_vx_second_switch_height is None) != (
        args.command_vx_after_second_switch is None
    ):
        parser.error(
            "--command-vx-second-switch-height and "
            "--command-vx-after-second-switch must be provided together"
        )
    if args.command_vx_second_switch_height is not None and switch_trigger_count == 0:
        parser.error("the second command switch requires a configured first command switch")
    if command_vx_schedule and (
        switch_trigger_count > 0 or args.command_vx_second_switch_height is not None
    ):
        parser.error(
            "--command-vx-schedule cannot be combined with the single/second "
            "command switch options"
        )

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
        record_video=args.record_video,
        video_path=args.video_path,
        video_fps=args.video_fps,
        video_width=args.video_width,
        video_height=args.video_height,
        video_camera=args.video_camera,
        video_follow_distance=args.video_follow_distance,
        video_follow_height=args.video_follow_height,
        video_follow_yaw_offset=args.video_follow_yaw_offset,
        video_follow_elevation=args.video_follow_elevation,
    )
    runner.command_vx_limit = args.command_vx_limit
    runner.command_vel[0] = np.clip(
        args.command_vx, -runner.command_vx_limit, runner.command_vx_limit
    )
    print(f"[INFO] Initial forward command: {runner.command_vel[0]:.2f} m/s")
    runner.auto_reset_on_flat_return = args.auto_reset_on_flat_return
    runner.auto_reset_climbed_height = args.auto_reset_climbed_height
    runner.auto_reset_flat_x_min = args.auto_reset_flat_x_min
    runner.auto_reset_flat_x_max = args.auto_reset_flat_x_max
    if args.auto_reset_on_flat_return:
        print(
            f"[INFO] Auto-reset on flat return enabled: arm at h >= "
            f"{args.auto_reset_climbed_height:.2f}m, flat when "
            f"x <= {args.auto_reset_flat_x_min:.2f}m or "
            f"x >= {args.auto_reset_flat_x_max:.2f}m"
        )
    runner.command_vx_switch_x = args.command_vx_switch_x
    runner.command_vx_switch_height = args.command_vx_switch_height
    runner.command_vx_switch_time = args.command_vx_switch_time
    runner.command_vx_after_switch = args.command_vx_after_switch
    runner.command_vx_switch_triggered = False
    runner.command_vx_second_switch_height = args.command_vx_second_switch_height
    runner.command_vx_after_second_switch = args.command_vx_after_second_switch
    runner.command_vx_second_switch_triggered = False
    runner.command_vx_schedule = command_vx_schedule
    runner.command_vx_schedule_index = 0
    if command_vx_schedule:
        print(
            "[INFO] Scheduled forward command stages: "
            + ", ".join(
                f"t >= {switch_time:.2f}s -> "
                f"{np.clip(velocity, -runner.command_vx_limit, runner.command_vx_limit):.2f} m/s"
                for switch_time, velocity in command_vx_schedule
            )
        )
    if args.command_vx_switch_x is not None:
        print(
            f"[INFO] Scheduled forward command switch: x >= {args.command_vx_switch_x:.2f} m "
            f"-> {np.clip(args.command_vx_after_switch, -runner.command_vx_limit, runner.command_vx_limit):.2f} m/s"
        )
    elif args.command_vx_switch_height is not None:
        print(
            f"[INFO] Scheduled forward command switch: base height >= "
            f"{args.command_vx_switch_height:.2f} m "
            f"-> {np.clip(args.command_vx_after_switch, -runner.command_vx_limit, runner.command_vx_limit):.2f} m/s"
        )
    elif args.command_vx_switch_time is not None:
        print(
            f"[INFO] Scheduled forward command switch: t >= "
            f"{args.command_vx_switch_time:.2f} s "
            f"-> {np.clip(args.command_vx_after_switch, -runner.command_vx_limit, runner.command_vx_limit):.2f} m/s"
        )
    if args.command_vx_second_switch_height is not None:
        print(
            f"[INFO] Scheduled second forward command switch: base height >= "
            f"{args.command_vx_second_switch_height:.2f} m "
            f"-> {np.clip(args.command_vx_after_second_switch, -runner.command_vx_limit, runner.command_vx_limit):.2f} m/s"
        )
    runner.run()


if __name__ == "__main__":
    main()
