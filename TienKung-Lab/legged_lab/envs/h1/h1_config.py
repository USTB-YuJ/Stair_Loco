# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import H1_CFG
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class H1RewardCfg(RewardCfg):
    # ===== 任务跟踪 (正向): 鼓励跟随速度命令 =====
    # 跟踪 xy 线速度命令: exp(-||vel_cmd - vel_actual||^2 / std^2)
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    # 跟踪 yaw 角速度命令, 同样 exp 形式
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})

    # ===== 姿态/稳定性 (负向): 抑制晃动和倾斜 =====
    # 抑制 base 的 z 方向线速度 (抑制蹦跳)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # 抑制 base 的 roll/pitch 角速度 (抑制身体晃动)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # base 倾斜惩罚 (用投影重力 xy 分量平方和度量)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # ===== 能耗 / 平滑性 (负向) =====
    # 关节能耗 ||torque * dq||
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    # 关节加速度平方和, 抗抖
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    # 相邻两步动作差平方和, 抗抖
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # ===== 接触约束 (负向) =====
    # 非脚部 (踝以外) 部位发生接触时扣分
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    # 双脚同时悬空 (无接触) 扣分, 防止跳跃步态
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"), "threshold": 1.0},
    )

    # ===== 终止信号 =====
    # 非超时终止时扣大分 (基于 is_terminated)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ===== 脚部行为 =====
    # 单脚支撑时奖励摆动腿空中时间 (上限 threshold=0.4s; 命令很小不奖励)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"), "threshold": 0.4},
    )
    # 触地状态下脚的水平速度 (滑步) 惩罚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
        },
    )
    # 脚部冲击力超过阈值时线性扣分 (软着地, 减少结构冲击)
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle.*"),
            "threshold": 800,
            "max_reward": 400,
        },
    )
    # 双脚距离过近时惩罚, 避免脚部互踩 / 异常步态
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle.*"]), "threshold": 0.3},
    )
    # 绊脚惩罚: 脚的水平接触力 > 5 倍法向接触力 时认为撞到了垂直障碍
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle.*"])},
    )

    # ===== 关节限位 / 偏离默认姿态 =====
    # 关节超出软限位 (默认 90%) 部分线性扣分
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    # 仅在“命令接近静止”时, 惩罚 hip yaw / roll 偏离默认姿态
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    )
    # 全时段 (含运动) 抑制手臂 (shoulder/elbow) 偏离默认姿态
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder.*", ".*_elbow.*"])},
    )
    # 全时段抑制腰 (torso_joint) 偏离默认姿态
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*torso_joint.*"])},
    )
    # 仅静止命令下抑制腿 (hip_pitch/knee/ankle) 偏离默认姿态
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*ankle.*"])},
    )


@configclass
class H1FlatEnvCfg(BaseEnvCfg):

    reward = H1RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = H1_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.feet_body_names = [".*ankle.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]


@configclass
class H1FlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "h1_flat"
    wandb_project: str = "h1_flat"


@configclass
class H1RoughEnvCfg(H1FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = -0.25


@configclass
class H1RoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "h1_rough"
    wandb_project: str = "h1_rough"

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"
