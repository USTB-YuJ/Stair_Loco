"""H1-2 task configs aligned with the g1 flat/rough stack."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import H1_2_CFG
from legged_lab.envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1RewardCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
)


@configclass
class H1_2RewardCfg(G1RewardCfg):
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


@configclass
class H1_2FlatEnvCfg(G1FlatEnvCfg):
    reward = H1_2RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = H1_2_CFG


@configclass
class H1_2FlatAgentCfg(G1FlatAgentCfg):
    experiment_name: str = "h1_2_flat"
    wandb_project: str = "h1_2_flat"


@configclass
class H1_2RoughEnvCfg(G1RoughEnvCfg):
    reward = H1_2RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = H1_2_CFG


@configclass
class H1_2RoughAgentCfg(G1RoughAgentCfg):
    experiment_name: str = "h1_2_rough"
    wandb_project: str = "h1_2_rough"
