"""Original H1 DWAQ configs.

Design choice:
- Reward stack is based on H1RewardCfg for robot-consistent shaping.
- Env/Agent keep DWAQ inheritance for framework compatibility.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import H1_CFG
from legged_lab.envs.g1.g1_dwaq_config import G1DwaqAgentCfg, G1DwaqEnvCfg
from legged_lab.envs.h1.h1_config import H1RewardCfg


@configclass
class H1DwaqRewardCfg(H1RewardCfg):
    """H1 reward baseline + DWAQ-specific terms."""

    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=3.0, params={"std": 0.4})
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_turn_relaxed,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"]),
            "command_threshold": 0.1,
            "active_command_scale": 0.0,
            "yaw_threshold": 0.25,
            "turn_command_scale": 0.0,
        },
    )

    energy = RewTerm(
        func=mdp.energy,
        weight=-1e-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    ".*_ankle_joint",
                ],
            )
        },
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    ".*_ankle_joint",
                ],
            )
        },
    )

    upright_posture_reward = RewTerm(
        func=mdp.body_orientation_exp,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"), "std": 0.3},
        weight=1.0,
    )
    alive = RewTerm(func=mdp.alive, weight=0.15)
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded_xy_yaw,
        weight=-2.0,
        params={
            "cmd_threshold": 0.2,
            "yaw_cmd_threshold": 0.2,
            "vel_threshold": 0.1,
            "yaw_vel_threshold": 0.1,
        },
    )
    target_goal_progress = RewTerm(
        func=mdp.target_goal_progress,
        weight=1.0,
        params={"min_progress_rate": -0.5, "max_progress_rate": 1.0},
    )
    target_goal_reached = RewTerm(func=mdp.target_goal_reached, weight=25.0)
    gait_phase_contact = RewTerm(
        func=mdp.terrain_scaled_gait_phase_contact,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["left_ankle.*", "right_ankle.*"]),
            "stance_threshold": 0.55,
            "stair_scale": 0.25,
            "turn_yaw_threshold": 0.35,
            "turn_scale": 0.5,
        },
    )
    feet_swing_height = RewTerm(
        func=mdp.terrain_aware_feet_swing_height,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
            "base_clearance": 0.08,
            "obstacle_clearance": 0.04,
            "max_clearance": 0.30,
            "high_tolerance": 0.12,
            "low_penalty_scale": 1.0,
            "high_penalty_scale": 0.25,
            "stance_threshold": 0.55,
        },
    )


@configclass
class H1DwaqEnvCfg(G1DwaqEnvCfg):
    reward = H1DwaqRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = H1_CFG
        self.robot.controlled_joint_names = [
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_joint",
        ]
        self.robot.fixed_joint_names = [
            ".*torso_joint.*",
            ".*_shoulder.*",
            ".*_elbow.*",
        ]
        self.domain_rand.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.robot.controlled_joint_names
        )
        self.robot.terminate_contacts_body_names = [".*torso.*", ".*pelvis.*", ".*_knee.*", ".*_elbow.*"]
        self.robot.termination_contact_force_threshold = 300.0
        # Enable tilt-based reset: terminate when body tilt exceeds 55 degrees.
        self.robot.termination_tilt_threshold_deg = 55.0
        self.robot.feet_body_names = ["left_ankle.*", "right_ankle.*"]
        self.commands.target_point.enable = True
        self.commands.target_point.goal_reached_radius = 0.5
        self.commands.target_point.goal_timeout_s = 14.0
        self.commands.target_point.terrain_margin = 1.0
        self.commands.target_point.min_goal_distance = 1.0
        self.commands.target_point.distance_to_max_speed = 2.0
        self.commands.target_point.terrain_speed_schedule_enable = True
        self.commands.target_point.terrain_speed_min_scale = 0.65
        self.commands.target_point.stair_vx_max = 0.55
        self.commands.target_point.stair_friendly_sampling_enable = True
        self.commands.target_point.stair_target_distance_range = (1.0, 2.5)
        self.commands.target_point.stair_target_lateral_range = (-0.4, 0.4)
        self.commands.target_point.yaw_only_probability = 0.2
        self.commands.target_point.arc_turn_probability = 0.3
        self.commands.target_point.yaw_only_ang_vel_z = (-0.8, 0.8)
        self.commands.target_point.yaw_only_min_abs_ang_vel_z = 0.25
        self.commands.target_point.arc_turn_lin_vel_x = (0.1, 0.5)
        self.commands.target_point.arc_turn_ang_vel_z = (-0.8, 0.8)
        self.commands.target_point.arc_turn_min_abs_ang_vel_z = 0.2
        self.commands.ranges.lin_vel_x = (0.0, self.commands.ranges.lin_vel_x[1])
        self.commands.ranges.lin_vel_y = (0.0, 0.0)


@configclass
class H1DwaqAgentCfg(G1DwaqAgentCfg):
    experiment_name: str = "h1_dwaq"
    wandb_project: str = "h1_dwaq"
