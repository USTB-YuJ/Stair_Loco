"""H1-2 DWAQ configs adapted from G1 DWAQ with H1-specific tuning.

H1-2 vs G1 physical differences that drive reward tuning:
  - Taller (init 1.05m vs 0.80m) → higher CoG, needs stronger orientation control
  - Lower joint damping (hip 2.5 vs 5.0) → less passive stability
  - Larger torque limits (200-300Nm vs 88-139Nm) → penalize energy/acc less per unit
  - More DoF (separate ankle pitch/roll, wrist joints) → larger action space

Tuning references H1 config (similar body size) while keeping DWAQ-specific rewards.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import H1_2_CFG
from legged_lab.envs.g1.g1_dwaq_config import G1DwaqAgentCfg, G1DwaqEnvCfg, G1DwaqRewardCfg


@configclass
class H1_2DwaqRewardCfg(G1DwaqRewardCfg):
    # ---- Smoothness penalties (reduced for H1-2's larger torque range) ----
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    energy = RewTerm(func=mdp.energy, weight=-5e-4)

    # ---- Orientation control (increased for H1-2's taller frame) ----
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
        weight=-3.0,
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    # ---- Bipedal gait (H1-sized robot needs stronger air-time incentive) ----
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.35,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.3},
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-2e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )

    # ---- Joint deviation (adapted to H1-2 joint structure) ----
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    )
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*torso.*",
                    ".*_shoulder_roll.*",
                    ".*_shoulder_yaw.*",
                    ".*_shoulder_pitch.*",
                    ".*_elbow.*",
                    ".*_wrist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*"])},
    )


@configclass
class H1_2DwaqEnvCfg(G1DwaqEnvCfg):
    reward = H1_2DwaqRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = H1_2_CFG
        self.robot.feet_body_names = ["left_ankle_roll.*", "right_ankle_roll.*"]


@configclass
class H1_2DwaqAgentCfg(G1DwaqAgentCfg):
    experiment_name: str = "h1_2_dwaq"
    wandb_project: str = "h1_2_dwaq"

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.8
