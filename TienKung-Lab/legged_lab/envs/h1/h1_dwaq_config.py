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

    # 直立姿态奖励 (正向): 朝向越接近直立, 奖励越接近 1
    # 形式: exp(-||torso 投影重力 xy||^2 / std^2), std=0.3
    upright_posture_reward = RewTerm(
        func=mdp.body_orientation_exp,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"), "std": 0.3},
        weight=1.0,
    )
    # 存活奖励: 每步常数 +0.15, 与 termination_penalty 形成平衡, 防止机器人“摆烂自杀”
    alive = RewTerm(func=mdp.alive, weight=0.15)
    # 偷懒惩罚: 收到运动命令但实际几乎不动 (|cmd|>0.2 且 |vel|<0.1) 的步骤会被扣分
    idle_penalty = RewTerm(
        func=mdp.idle_when_commanded,
        weight=-2.0,
        params={"cmd_threshold": 0.2, "vel_threshold": 0.1},
    )
    # 步态相位匹配奖励: 双脚接触状态符合期望 stance/swing 相位时 +1
    # body_names 顺序需与 leg_phase 对齐: [左脚, 右脚]
    gait_phase_contact = RewTerm(
        func=mdp.gait_phase_contact,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["left_ankle.*", "right_ankle.*"]),
            "stance_threshold": 0.55,
        },
    )
    # 摆动脚 "抬不够" 惩罚 (地形相对 + 单边惩罚 + 自适应)
    # - clearance = feet_z - terrain_z_under_foot (使用支撑脚下方地形)
    # - 自适应: target 会随前方障碍高度自动加大, 鼓励上台阶时多抬脚
    # - 单边: 只罚 “抬太低”, 抬高一点不罚, 不会和上台阶冲突
    feet_swing_height = RewTerm(
        func=mdp.feet_swing_clearance,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["left_ankle.*", "right_ankle.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle.*", "right_ankle.*"]),
            "target_clearance": 0.18,
            "adaptive": True,
            "extra_clearance": 0.05,
            "max_target": 0.40,
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
