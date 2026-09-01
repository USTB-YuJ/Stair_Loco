"""H1 DWAQ configuration for the task-specific three-step corridor."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.envs.h1.h1_dwaq_config import (
    H1DwaqAgentCfg,
    H1DwaqEnvCfg,
    H1DwaqRewardCfg,
)
from legged_lab.terrains import THREE_STEP_CORRIDOR_TERRAINS_CFG


@configclass
class H1ThreeStepTaskCfg:
    """Task geometry, command, and termination parameters."""

    forward_velocity_range: tuple[float, float] = (0.7, 1.0)
    turn_yaw_abs_range: tuple[float, float] = (0.4, 0.8)
    approach_distance: float = 1.0
    flat_forward_success_distance: float = 2.18
    lateral_deadband: float = 0.15
    lateral_exit_distance: float = 0.65
    backward_exit_distance: float = 0.30
    max_forward_progress_velocity: float = 1.5


@configclass
class H1ThreeStepRewardCfg(H1DwaqRewardCfg):
    """Existing H1 DWAQ shaping plus task-completion terms."""

    three_step_success = RewTerm(func=mdp.three_step_success, weight=500.0)
    forward_progress = RewTerm(func=mdp.three_step_forward_progress, weight=1.0)
    corridor_lateral_deviation = RewTerm(
        func=mdp.three_step_lateral_deviation,
        weight=-2.0,
    )
    corridor_heading_deviation = RewTerm(
        func=mdp.three_step_heading_deviation,
        weight=-1.0,
    )
    termination_penalty = RewTerm(func=mdp.three_step_failed_termination, weight=-200.0)


@configclass
class H1ThreeStepDwaqEnvCfg(H1DwaqEnvCfg):
    """Independent H1 task for blind traversal of three ascending steps."""

    reward = H1ThreeStepRewardCfg()
    three_step_task = H1ThreeStepTaskCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_generator = THREE_STEP_CORRIDOR_TERRAINS_CFG
        self.scene.max_init_terrain_level = 0
        self.scene.max_episode_length_s = 10.0

        self.commands.heading_command = False
        self.commands.rel_heading_envs = 0.0
        self.commands.rel_standing_envs = 0.0
        self.commands.debug_vis = False
        self.commands.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.ranges.lin_vel_x = self.three_step_task.forward_velocity_range
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (
            -self.three_step_task.turn_yaw_abs_range[1],
            self.three_step_task.turn_yaw_abs_range[1],
        )
        self.commands.ranges.heading = None

        self.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "yaw": (-math.radians(5.0), math.radians(5.0)),
        }
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            axis: (-0.2, 0.2)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.domain_rand.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class H1ThreeStepDwaqAgentCfg(H1DwaqAgentCfg):
    experiment_name: str = "h1_dwaq_three_step"
    wandb_project: str = "h1_dwaq_three_step"

    def __post_init__(self):
        super().__post_init__()
        # This is a task-transfer resume: keep the policy weights, but use a
        # conservative exploration level for the new fixed stair distribution.
        self.policy.resume_noise_std = 0.2
        # The source run used a large PPO update budget.  A smaller transfer
        # step prevents the VAE/critic from chasing the new task distribution
        # before the policy has re-adapted to the corridor.
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.num_learning_epochs = 1
