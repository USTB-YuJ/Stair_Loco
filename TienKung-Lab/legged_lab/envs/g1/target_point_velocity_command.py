"""Target-point velocity command for DWAQ locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG


class TargetPointVelocityCommand(UniformVelocityCommand):
    """Generate forward-only velocity commands from random terrain-local targets.

    The public command interface stays identical to UniformVelocityCommand:
    command[:, 0] is forward velocity, command[:, 1] is always zero, and
    command[:, 2] is yaw velocity in the robot base frame.
    """

    def __init__(self, cfg, env, target_cfg):
        self.target_cfg = target_cfg
        super().__init__(cfg, env)

        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_age = torch.zeros(self.num_envs, device=self.device)
        self.goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.goal_heading_error = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_heading_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TargetPointVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tGoal timeout: {self.target_cfg.goal_timeout_s}\n"
        msg += f"\tGoal reached radius: {self.target_cfg.goal_reached_radius}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def _env_ids_to_tensor(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, device=self.device, dtype=torch.long)

    def _resample(self, env_ids: Sequence[int] | slice):
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        self.time_left[env_ids] = float(self.target_cfg.goal_timeout_s)
        self._resample_command(env_ids)
        self.command_counter[env_ids] += 1

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        self.target_age[env_ids] = 0.0
        self.is_standing_env[env_ids] = (
            torch.rand(env_ids.numel(), device=self.device) <= self.cfg.rel_standing_envs
        )
        self._sample_targets_in_current_terrain(env_ids)
        self._refresh_goal_error(env_ids)
        self._set_velocity_command(env_ids)

    def _update_metrics(self):
        self._refresh_goal_error()
        timeout = max(float(self.target_cfg.goal_timeout_s), self._env.step_dt)
        max_command_step = timeout / self._env.step_dt
        self.metrics["goal_distance"] += self.goal_distance / max_command_step
        self.metrics["goal_heading_error"] += torch.abs(self.goal_heading_error) / max_command_step
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _update_command(self):
        self.target_age += self._env.step_dt
        self._refresh_goal_error()

        reached_env_ids = (
            (self.goal_distance <= self.target_cfg.goal_reached_radius) & (~self.is_standing_env)
        ).nonzero(as_tuple=False).flatten()
        if reached_env_ids.numel() > 0:
            self._resample(reached_env_ids)
            self._refresh_goal_error()

        self._set_velocity_command()

    def _sample_targets_in_current_terrain(self, env_ids: torch.Tensor):
        root_xy = self.robot.data.root_pos_w[env_ids, :2]
        origins_xy = self._env.scene.env_origins[env_ids, :2]
        half_extent = self._terrain_half_extent()
        min_distance = float(self.target_cfg.min_goal_distance)

        target_xy = self._sample_xy(origins_xy, half_extent)
        valid = torch.linalg.norm(target_xy - root_xy, dim=-1) >= min_distance
        for _ in range(32):
            if torch.all(valid):
                break
            invalid_ids = (~valid).nonzero(as_tuple=False).flatten()
            target_xy[invalid_ids] = self._sample_xy(origins_xy[invalid_ids], half_extent)
            valid = torch.linalg.norm(target_xy - root_xy, dim=-1) >= min_distance

        self.target_pos_w[env_ids, :2] = target_xy
        self.target_pos_w[env_ids, 2] = self._env.scene.env_origins[env_ids, 2] + 0.15

    def _terrain_half_extent(self) -> torch.Tensor:
        terrain_generator = getattr(self._env.scene.terrain.cfg, "terrain_generator", None)
        if terrain_generator is not None and getattr(terrain_generator, "size", None) is not None:
            size_x, size_y = terrain_generator.size
        else:
            scene_cfg = getattr(self._env.scene, "cfg", None)
            size_x = size_y = getattr(scene_cfg, "env_spacing", 8.0)
        margin = float(self.target_cfg.terrain_margin)
        half_x = max(float(size_x) * 0.5 - margin, 0.0)
        half_y = max(float(size_y) * 0.5 - margin, 0.0)
        return torch.tensor([half_x, half_y], device=self.device)

    def _sample_xy(self, origins_xy: torch.Tensor, half_extent: torch.Tensor) -> torch.Tensor:
        unit = torch.empty(origins_xy.shape, device=self.device).uniform_(-1.0, 1.0)
        return origins_xy + unit * half_extent

    def _refresh_goal_error(self, env_ids: Sequence[int] | slice | None = None):
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = self._env_ids_to_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        delta_xy = self.target_pos_w[env_ids_tensor, :2] - self.robot.data.root_pos_w[env_ids_tensor, :2]
        distance = torch.linalg.norm(delta_xy, dim=-1)
        target_heading = torch.atan2(delta_xy[:, 1], delta_xy[:, 0])
        heading_error = math_utils.wrap_to_pi(target_heading - self.robot.data.heading_w[env_ids_tensor])
        self.goal_distance[env_ids_tensor] = distance
        self.goal_heading_error[env_ids_tensor] = heading_error

    def _set_velocity_command(self, env_ids: Sequence[int] | slice | None = None):
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = self._env_ids_to_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        distance = self.goal_distance[env_ids_tensor]
        heading_error = self.goal_heading_error[env_ids_tensor]
        vx_max = max(float(self.cfg.ranges.lin_vel_x[1]), 0.0)
        distance_scale = torch.clamp(distance / float(self.target_cfg.distance_to_max_speed), 0.0, 1.0)
        turn_gate = torch.clamp(torch.cos(heading_error), 0.0, 1.0)
        yaw_rate = torch.clamp(
            self.cfg.heading_control_stiffness * heading_error,
            min=float(self.cfg.ranges.ang_vel_z[0]),
            max=float(self.cfg.ranges.ang_vel_z[1]),
        )

        self.vel_command_b[env_ids_tensor, 0] = vx_max * distance_scale * turn_gate
        self.vel_command_b[env_ids_tensor, 1] = 0.0
        self.vel_command_b[env_ids_tensor, 2] = yaw_rate

        standing_env_ids = env_ids_tensor[self.is_standing_env[env_ids_tensor]]
        if standing_env_ids.numel() > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "target_pos_visualizer"):
                target_marker_cfg = POSITION_GOAL_MARKER_CFG.replace(prim_path="/Visuals/Command/target_point")
                self.target_pos_visualizer = VisualizationMarkers(target_marker_cfg)
            self.target_pos_visualizer.set_visibility(True)
        elif hasattr(self, "target_pos_visualizer"):
            self.target_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if not self.robot.is_initialized or not hasattr(self, "target_pos_visualizer"):
            return
        scales = torch.ones_like(self.target_pos_w) * 8.0
        marker_indices = torch.where(
            self.is_standing_env,
            torch.full_like(self.is_standing_env, 2, dtype=torch.long),
            torch.zeros_like(self.is_standing_env, dtype=torch.long),
        )
        self.target_pos_visualizer.visualize(self.target_pos_w, scales=scales, marker_indices=marker_indices)
