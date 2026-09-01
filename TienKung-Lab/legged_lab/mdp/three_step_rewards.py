"""Reward terms for the H1 three-step corridor task."""

from __future__ import annotations

import torch

from isaaclab.utils import math as math_utils


def three_step_success(env) -> torch.Tensor:
    """Return one on the single step where a goal-bearing episode succeeds."""
    return env.three_step_success_buf.float()


def three_step_forward_progress(env) -> torch.Tensor:
    """Reward positive corridor progress for stairs and flat-forward tasks."""
    return env.three_step_forward_progress_vel * env.three_step_forward_mask.float()


def three_step_lateral_deviation(env) -> torch.Tensor:
    """Penalize squared lateral displacement outside the corridor deadband."""
    lateral_offset = torch.abs(env.robot.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1])
    excess = torch.clamp(lateral_offset - env.cfg.three_step_task.lateral_deadband, min=0.0)
    return torch.square(excess)


def three_step_heading_deviation(env) -> torch.Tensor:
    """Keep forward tasks aligned with the corridor's world +x direction."""
    heading_error = math_utils.wrap_to_pi(env.robot.data.heading_w)
    return torch.square(heading_error) * env.three_step_forward_mask.float()


def three_step_failed_termination(env) -> torch.Tensor:
    """Penalize failures while excluding both timeouts and successful exits."""
    return (
        env.reset_buf
        & ~env.time_out_buf
        & ~env.three_step_success_buf
    ).float()

