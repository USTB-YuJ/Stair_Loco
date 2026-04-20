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

    upright_posture_reward = RewTerm(
        func=mdp.body_orientation_exp,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"), "std": 0.3},
        weight=2.0,
    )
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


@configclass
class H1DwaqEnvCfg(G1DwaqEnvCfg):
    reward = H1DwaqRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = H1_CFG
        self.robot.terminate_contacts_body_names = [".*torso.*", ".*pelvis.*", ".*_knee.*", ".*_elbow.*"]
        self.robot.termination_contact_force_threshold = 500.0
        # Enable tilt-based reset: terminate when body tilt exceeds 55 degrees.
        self.robot.termination_tilt_threshold_deg = 55.0
        self.robot.feet_body_names = ["left_ankle.*", "right_ankle.*"]


@configclass
class H1DwaqAgentCfg(G1DwaqAgentCfg):
    experiment_name: str = "h1_dwaq"
    wandb_project: str = "h1_dwaq"
