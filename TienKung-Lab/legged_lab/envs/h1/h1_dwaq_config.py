"""Original H1 DWAQ configs.

Design choice:
- Reward stack is based on H1RewardCfg for robot-consistent shaping.
- Env/Agent keep DWAQ inheritance for framework compatibility.
"""

from isaaclab.envs.mdp import events as isaaclab_events
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import H1_PAYLOAD_HORIZONTAL_CFG
from legged_lab.envs.g1.g1_dwaq_config import G1DwaqAgentCfg, G1DwaqEnvCfg
from legged_lab.envs.h1.h1_config import H1RewardCfg


@configclass
class H1DwaqRewardCfg(H1RewardCfg):
    """H1 reward baseline + DWAQ-specific terms."""

    upright_posture_reward = RewTerm(
        func=mdp.body_orientation_exp,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"), "std": 0.3},
        weight=2.5,
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    alive = RewTerm(func=mdp.alive, weight=0.15)
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded,
        weight=-2.0,
        params={"cmd_threshold": 0.2, "vel_threshold": 0.1},
    )
    gait_phase_contact = RewTerm(
        func=mdp.gait_phase_contact,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["left_ankle.*", "right_ankle.*"]),
            "stance_threshold": 0.55,
        },
    )
    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
            "target_height": 0.08,
        },
    )
    terrain_aware_feet_clearance = RewTerm(
        func=mdp.terrain_aware_feet_clearance,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
            "base_clearance": 0.08,
            "obstacle_margin": 0.04,
            "max_clearance": 0.30,
            "obstacle_threshold": 0.02,
            "under_clearance_std": 0.04,
            "overshoot_margin": 0.08,
            "overshoot_std": 0.08,
        },
    )


@configclass
class H1DwaqEnvCfg(G1DwaqEnvCfg):
    reward = H1DwaqRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = H1_PAYLOAD_HORIZONTAL_CFG
        self.robot.controlled_joint_names = [
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
        self.robot.payload_mount_joint_names = [
            "payload_mount_x_joint",
            "payload_mount_y_joint",
            "payload_mount_z_joint",
            "payload_mount_roll_joint",
            "payload_mount_pitch_joint",
            "payload_mount_yaw_joint",
        ]
        self.robot.payload_mount_joint_ranges = {
            "payload_mount_x_joint": (-0.03, 0.03),
            "payload_mount_y_joint": (-0.02, 0.02),
            "payload_mount_z_joint": (-0.02, 0.02),
            "payload_mount_roll_joint": (-0.12, 0.12),
            "payload_mount_pitch_joint": (-0.12, 0.12),
            "payload_mount_yaw_joint": (-0.12, 0.12),
        }
        self.domain_rand.events.payload_mass = EventTerm(
            func=isaaclab_events.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="h1_extinguisher_payload"),
                "mass_distribution_params": (2.0, 4.0),
                "operation": "abs",
            },
        )
        self.domain_rand.events.payload_mount_pose = EventTerm(
            func=mdp.randomize_payload_mount_joints,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=self.robot.payload_mount_joint_names,
                    preserve_order=True,
                ),
                "joint_ranges": self.robot.payload_mount_joint_ranges,
            },
        )
        self.robot.terminate_contacts_body_names = [".*torso.*", ".*pelvis.*", ".*_knee.*", ".*_elbow.*"]
        self.robot.termination_contact_force_threshold = 300.0
        # Enable tilt-based reset: terminate when body tilt exceeds 55 degrees.
        self.robot.termination_tilt_threshold_deg = 55.0
        self.robot.feet_body_names = ["left_ankle.*", "right_ankle.*"]


@configclass
class H1DwaqAgentCfg(G1DwaqAgentCfg):
    experiment_name: str = "h1_dwaq"
    wandb_project: str = "h1_dwaq"
