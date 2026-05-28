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

    MODE_TARGET = 0
    MODE_YAW_ONLY = 1
    MODE_ARC_TURN = 2

    def __init__(self, cfg, env, target_cfg):
        self.target_cfg = target_cfg
        super().__init__(cfg, env)

        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_age = torch.zeros(self.num_envs, device=self.device)
        self.goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.goal_progress = torch.zeros(self.num_envs, device=self.device)
        self.goal_heading_error = torch.zeros(self.num_envs, device=self.device)
        self.goal_reached_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_timed_out_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.command_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.direct_command_b = torch.zeros(self.num_envs, 3, device=self.device)

        self.curriculum_window_size = max(int(self.target_cfg.curriculum_window_size), 1)
        self.target_success_history = torch.zeros(
            self.num_envs, self.curriculum_window_size, dtype=torch.bool, device=self.device
        )
        self.target_timeout_history = torch.zeros(
            self.num_envs, self.curriculum_window_size, dtype=torch.bool, device=self.device
        )
        self.target_history_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_attempt_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_reached_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_timeout_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_heading_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["scheduled_vx_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_mode_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["yaw_only_mode_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["arc_turn_mode_rate"] = torch.zeros(self.num_envs, device=self.device)

        self._terrain_col_names = self._build_terrain_col_names()
        self._terrain_col_is_stairs = self._terrain_col_mask("stairs")
        self._terrain_col_is_stairs_up = self._terrain_col_mask("stairs_up")
        self._terrain_col_is_stairs_down = self._terrain_col_mask("stairs_down")

    def __str__(self) -> str:
        msg = "TargetPointVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tGoal timeout: {self.target_cfg.goal_timeout_s}\n"
        msg += f"\tGoal reached radius: {self.target_cfg.goal_reached_radius}\n"
        msg += f"\tYaw-only probability: {self.target_cfg.yaw_only_probability}\n"
        msg += f"\tArc-turn probability: {self.target_cfg.arc_turn_probability}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def _terrain_generator_cfg(self):
        terrain = getattr(self._env.scene, "terrain", None)
        terrain_cfg = getattr(terrain, "cfg", None)
        return getattr(terrain_cfg, "terrain_generator", None)

    def _build_terrain_col_names(self) -> list[str]:
        terrain_generator = self._terrain_generator_cfg()
        sub_terrains = getattr(terrain_generator, "sub_terrains", None)
        num_cols = int(getattr(terrain_generator, "num_cols", 0) or 0)
        if not sub_terrains or num_cols <= 0:
            return []

        names = list(sub_terrains.keys())
        proportions = [max(float(getattr(sub_terrains[name], "proportion", 0.0)), 0.0) for name in names]
        total = sum(proportions)
        if total <= 0.0:
            proportions = [1.0 / len(names)] * len(names)
        else:
            proportions = [value / total for value in proportions]

        cumulative = []
        running = 0.0
        for value in proportions:
            running += value
            cumulative.append(running)

        col_names = []
        for col in range(num_cols):
            ratio = col / num_cols + 0.001
            index = len(cumulative) - 1
            for candidate, threshold in enumerate(cumulative):
                if ratio < threshold:
                    index = candidate
                    break
            col_names.append(names[index])
        return col_names

    def _terrain_col_mask(self, pattern: str) -> torch.Tensor:
        if not self._terrain_col_names:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        return torch.tensor(
            [pattern in name for name in self._terrain_col_names], dtype=torch.bool, device=self.device
        )

    def _terrain_col_ids(self, env_ids: Sequence[int] | slice) -> torch.Tensor | None:
        env_ids = self._env_ids_to_tensor(env_ids)
        terrain = getattr(self._env.scene, "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is None or self._terrain_col_is_stairs.numel() == 0:
            return None
        max_col = self._terrain_col_is_stairs.numel() - 1
        return torch.clamp(terrain_types[env_ids].to(dtype=torch.long), min=0, max=max_col)

    def _stair_masks(self, env_ids: Sequence[int] | slice) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        env_ids = self._env_ids_to_tensor(env_ids)
        col_ids = self._terrain_col_ids(env_ids)
        if col_ids is None:
            empty = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
            return empty, empty, empty
        return (
            self._terrain_col_is_stairs[col_ids],
            self._terrain_col_is_stairs_up[col_ids],
            self._terrain_col_is_stairs_down[col_ids],
        )

    def get_terrain_category_masks(self, env_ids: Sequence[int] | slice) -> dict[str, torch.Tensor]:
        env_ids = self._env_ids_to_tensor(env_ids)
        stairs, stairs_up, stairs_down = self._stair_masks(env_ids)
        return {
            "stairs": stairs,
            "stairs_up": stairs_up,
            "stairs_down": stairs_down,
            "non_stairs": ~stairs,
        }

    def get_target_timeout_rate(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() == 0:
            return torch.zeros(0, device=self.device)
        attempts = self.target_attempt_count[env_ids].clamp(min=1).float()
        timeouts = self.target_timeout_history[env_ids].float().sum(dim=-1)
        return timeouts / attempts

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
        self.command_mode[env_ids] = self.MODE_TARGET
        self.direct_command_b[env_ids] = 0.0
        self.is_standing_env[env_ids] = (
            torch.rand(env_ids.numel(), device=self.device) <= self.cfg.rel_standing_envs
        )

        active_env_ids = env_ids[~self.is_standing_env[env_ids]]
        if active_env_ids.numel() > 0:
            self._sample_command_modes(active_env_ids)

        target_env_ids = env_ids[self.command_mode[env_ids] == self.MODE_TARGET]
        if target_env_ids.numel() > 0:
            self._sample_targets_in_current_terrain(target_env_ids)
            self._refresh_goal_error(target_env_ids)
            self.previous_goal_distance[target_env_ids] = self.goal_distance[target_env_ids]

        direct_env_ids = env_ids[self.command_mode[env_ids] != self.MODE_TARGET]
        if direct_env_ids.numel() > 0:
            self.target_pos_w[direct_env_ids, :2] = self.robot.data.root_pos_w[direct_env_ids, :2]
            self.target_pos_w[direct_env_ids, 2] = self.robot.data.root_pos_w[direct_env_ids, 2] + 0.15
            self.goal_distance[direct_env_ids] = 0.0
            self.goal_heading_error[direct_env_ids] = 0.0
            self.previous_goal_distance[direct_env_ids] = 0.0

        self._set_velocity_command(env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() > 0:
            self.goal_progress[env_ids] = 0.0
            self.goal_reached_this_step[env_ids] = False
            self.goal_timed_out_this_step[env_ids] = False
            self.previous_goal_distance[env_ids] = self.goal_distance[env_ids]
        return extras

    def _clamped_probability(self, name: str) -> float:
        return min(max(float(getattr(self.target_cfg, name, 0.0)), 0.0), 1.0)

    def _sample_command_modes(self, env_ids: torch.Tensor):
        yaw_prob = self._clamped_probability("yaw_only_probability")
        arc_prob = self._clamped_probability("arc_turn_probability")
        turn_prob_sum = yaw_prob + arc_prob
        if turn_prob_sum > 1.0:
            yaw_prob /= turn_prob_sum
            arc_prob /= turn_prob_sum

        mode_sample = torch.rand(env_ids.numel(), device=self.device)
        yaw_mask = mode_sample < yaw_prob
        arc_mask = (mode_sample >= yaw_prob) & (mode_sample < yaw_prob + arc_prob)

        yaw_env_ids = env_ids[yaw_mask]
        if yaw_env_ids.numel() > 0:
            self.command_mode[yaw_env_ids] = self.MODE_YAW_ONLY
            self.direct_command_b[yaw_env_ids, 0] = 0.0
            self.direct_command_b[yaw_env_ids, 1] = 0.0
            self.direct_command_b[yaw_env_ids, 2] = self._sample_signed_uniform(
                yaw_env_ids.numel(),
                self.target_cfg.yaw_only_ang_vel_z,
                float(self.target_cfg.yaw_only_min_abs_ang_vel_z),
            )

        arc_env_ids = env_ids[arc_mask]
        if arc_env_ids.numel() > 0:
            lin_range = self.target_cfg.arc_turn_lin_vel_x
            self.command_mode[arc_env_ids] = self.MODE_ARC_TURN
            self.direct_command_b[arc_env_ids, 0] = torch.empty(arc_env_ids.numel(), device=self.device).uniform_(
                float(lin_range[0]), float(lin_range[1])
            )
            self.direct_command_b[arc_env_ids, 1] = 0.0
            self.direct_command_b[arc_env_ids, 2] = self._sample_signed_uniform(
                arc_env_ids.numel(),
                self.target_cfg.arc_turn_ang_vel_z,
                float(self.target_cfg.arc_turn_min_abs_ang_vel_z),
            )

    def _sample_signed_uniform(self, count: int, value_range: tuple, min_abs: float) -> torch.Tensor:
        low = float(value_range[0])
        high = float(value_range[1])
        if high < low:
            low, high = high, low
        samples = torch.empty(count, device=self.device).uniform_(low, high)
        min_abs = max(float(min_abs), 0.0)
        max_abs = max(abs(low), abs(high))
        if min_abs <= 0.0 or max_abs < min_abs:
            return samples

        small = torch.abs(samples) < min_abs
        if torch.any(small):
            num_small = int(torch.sum(small).item())
            signs = torch.where(
                torch.rand(num_small, device=self.device) < 0.5,
                torch.full((num_small,), -1.0, device=self.device),
                torch.ones(num_small, device=self.device),
            )
            if low >= 0.0:
                signs = torch.ones_like(signs)
            elif high <= 0.0:
                signs = -torch.ones_like(signs)
            magnitudes = torch.empty(num_small, device=self.device).uniform_(min_abs, max_abs)
            samples[small] = torch.clamp(signs * magnitudes, min=low, max=high)
        return samples

    def compute(self, dt: float):
        self._update_metrics()
        self.goal_progress.zero_()
        self.goal_reached_this_step.zero_()
        self.goal_timed_out_this_step.zero_()

        self.target_age += dt
        self.time_left -= dt
        self._refresh_goal_error(self._active_non_standing_env_ids(slice(None), require_target=True))

        active_env_ids = self._active_non_standing_env_ids(slice(None), require_target=True)
        if active_env_ids.numel() > 0:
            self.goal_progress[active_env_ids] = (
                self.previous_goal_distance[active_env_ids] - self.goal_distance[active_env_ids]
            )
            self.previous_goal_distance[active_env_ids] = self.goal_distance[active_env_ids]

        self._update_command()

        timeout_env_ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if timeout_env_ids.numel() > 0:
            self._record_timeout_failures(timeout_env_ids)
            self._resample(timeout_env_ids)

        self._set_velocity_command()

    def record_reset_failures(self, env_ids: Sequence[int] | slice):
        env_ids = self._active_non_standing_env_ids(
            env_ids, min_age=float(self.target_cfg.min_reset_failure_age), require_target=True
        )
        if env_ids.numel() > 0:
            self._record_target_attempt(env_ids, success=False, timed_out=False)

    def get_terrain_curriculum_decisions(self, env_ids: Sequence[int] | slice) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() == 0:
            empty = torch.zeros(0, dtype=torch.bool, device=self.device)
            return empty, empty

        attempts = self.target_attempt_count[env_ids]
        successes = self.target_success_history[env_ids].long().sum(dim=-1)
        enough_attempts = attempts >= int(self.target_cfg.curriculum_min_attempts)
        move_up = enough_attempts & (successes >= int(self.target_cfg.curriculum_move_up_successes))
        move_down = enough_attempts & (successes <= int(self.target_cfg.curriculum_move_down_successes))
        move_down &= ~move_up
        return move_up, move_down

    def get_target_success_rate(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() == 0:
            return torch.zeros(0, device=self.device)

        attempts = self.target_attempt_count[env_ids].clamp(min=1).float()
        successes = self.target_success_history[env_ids].float().sum(dim=-1)
        return successes / attempts

    def clear_curriculum_history(self, env_ids: Sequence[int] | slice):
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        self.target_success_history[env_ids] = False
        self.target_timeout_history[env_ids] = False
        self.target_history_index[env_ids] = 0
        self.target_attempt_count[env_ids] = 0

    def _record_timeout_failures(self, env_ids: Sequence[int] | slice):
        env_ids = self._active_non_standing_env_ids(env_ids, require_target=True)
        if env_ids.numel() > 0:
            self.goal_timed_out_this_step[env_ids] = True
            self._record_target_attempt(env_ids, success=False, timed_out=True)

    def _record_target_attempt(self, env_ids: Sequence[int] | slice, success: bool, timed_out: bool = False):
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        history_indices = self.target_history_index[env_ids]
        self.target_success_history[env_ids, history_indices] = success
        self.target_timeout_history[env_ids, history_indices] = timed_out
        self.target_history_index[env_ids] = (history_indices + 1) % self.curriculum_window_size
        self.target_attempt_count[env_ids] = torch.clamp(
            self.target_attempt_count[env_ids] + 1, max=self.curriculum_window_size
        )

    def _active_non_standing_env_ids(
        self, env_ids: Sequence[int] | slice, min_age: float = 0.0, require_target: bool = False
    ) -> torch.Tensor:
        env_ids = self._env_ids_to_tensor(env_ids)
        if env_ids.numel() == 0:
            return env_ids
        active_mask = (self.target_age[env_ids] >= min_age) & (~self.is_standing_env[env_ids])
        if require_target:
            active_mask &= self.command_mode[env_ids] == self.MODE_TARGET
        return env_ids[active_mask]

    def _update_metrics(self):
        target_env_ids = self._active_non_standing_env_ids(slice(None), require_target=True)
        self._refresh_goal_error(target_env_ids)
        timeout = max(float(self.target_cfg.goal_timeout_s), self._env.step_dt)
        max_command_step = timeout / self._env.step_dt
        if target_env_ids.numel() > 0:
            self.metrics["goal_distance"][target_env_ids] += self.goal_distance[target_env_ids] / max_command_step
            self.metrics["goal_progress"][target_env_ids] += self.goal_progress[target_env_ids] / max_command_step
            self.metrics["goal_reached_rate"][target_env_ids] += (
                self.goal_reached_this_step[target_env_ids].float() / max_command_step
            )
            self.metrics["goal_timeout_rate"][target_env_ids] += (
                self.goal_timed_out_this_step[target_env_ids].float() / max_command_step
            )
            self.metrics["goal_heading_error"][target_env_ids] += (
                torch.abs(self.goal_heading_error[target_env_ids]) / max_command_step
            )
        active_mask = ~self.is_standing_env
        self.metrics["target_mode_rate"] += (
            (active_mask & (self.command_mode == self.MODE_TARGET)).float() / max_command_step
        )
        self.metrics["yaw_only_mode_rate"] += (
            (active_mask & (self.command_mode == self.MODE_YAW_ONLY)).float() / max_command_step
        )
        self.metrics["arc_turn_mode_rate"] += (
            (active_mask & (self.command_mode == self.MODE_ARC_TURN)).float() / max_command_step
        )
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _update_command(self):
        self._refresh_goal_error()

        reached_env_ids = (
            (self.goal_distance <= self.target_cfg.goal_reached_radius)
            & (~self.is_standing_env)
            & (self.command_mode == self.MODE_TARGET)
        ).nonzero(as_tuple=False).flatten()
        if reached_env_ids.numel() > 0:
            self.goal_reached_this_step[reached_env_ids] = True
            self._record_target_attempt(reached_env_ids, success=True, timed_out=False)
            self._resample(reached_env_ids)
            self._refresh_goal_error(reached_env_ids)

    def _sample_targets_in_current_terrain(self, env_ids: torch.Tensor):
        root_xy = self.robot.data.root_pos_w[env_ids, :2]
        origins_xy = self._env.scene.env_origins[env_ids, :2]
        half_extent = self._terrain_half_extent()
        min_distance = float(self.target_cfg.min_goal_distance)

        target_xy = self._sample_target_xy_for_envs(env_ids, origins_xy, half_extent)
        valid = torch.linalg.norm(target_xy - root_xy, dim=-1) >= min_distance
        for _ in range(32):
            if torch.all(valid):
                break
            invalid_local_ids = (~valid).nonzero(as_tuple=False).flatten()
            invalid_env_ids = env_ids[invalid_local_ids]
            target_xy[invalid_local_ids] = self._sample_target_xy_for_envs(
                invalid_env_ids, origins_xy[invalid_local_ids], half_extent
            )
            valid = torch.linalg.norm(target_xy - root_xy, dim=-1) >= min_distance

        if not torch.all(valid):
            invalid_local_ids = (~valid).nonzero(as_tuple=False).flatten()
            target_xy[invalid_local_ids] = self._sample_xy(origins_xy[invalid_local_ids], half_extent)

        self.target_pos_w[env_ids, :2] = target_xy
        self.target_pos_w[env_ids, 2] = self._env.scene.env_origins[env_ids, 2] + 0.15

    def _sample_target_xy_for_envs(
        self, env_ids: torch.Tensor, origins_xy: torch.Tensor, half_extent: torch.Tensor
    ) -> torch.Tensor:
        target_xy = self._sample_xy(origins_xy, half_extent)
        if not bool(self.target_cfg.stair_friendly_sampling_enable):
            return target_xy

        stairs_mask, _, _ = self._stair_masks(env_ids)
        if torch.any(stairs_mask):
            stair_env_ids = env_ids[stairs_mask]
            target_xy[stairs_mask] = self._sample_stair_friendly_xy(
                stair_env_ids, origins_xy[stairs_mask], half_extent
            )
        return target_xy

    def _sample_stair_friendly_xy(
        self, env_ids: torch.Tensor, origins_xy: torch.Tensor, half_extent: torch.Tensor
    ) -> torch.Tensor:
        distance_range = self.target_cfg.stair_target_distance_range
        lateral_range = self.target_cfg.stair_target_lateral_range
        distance = torch.empty(env_ids.numel(), device=self.device).uniform_(
            float(distance_range[0]), float(distance_range[1])
        )
        lateral = torch.empty(env_ids.numel(), device=self.device).uniform_(
            float(lateral_range[0]), float(lateral_range[1])
        )

        heading = self.robot.data.heading_w[env_ids]
        forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
        left = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=-1)
        target_xy = self.robot.data.root_pos_w[env_ids, :2] + forward * distance.unsqueeze(-1)
        target_xy += left * lateral.unsqueeze(-1)

        min_xy = origins_xy - half_extent
        max_xy = origins_xy + half_extent
        return torch.max(torch.min(target_xy, max_xy), min_xy)

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

    def _scheduled_vx_max(self, env_ids: torch.Tensor) -> torch.Tensor:
        base_vx_max = max(float(self.cfg.ranges.lin_vel_x[1]), 0.0)
        vx_max = torch.full((env_ids.numel(),), base_vx_max, device=self.device)
        if not bool(self.target_cfg.terrain_speed_schedule_enable):
            return vx_max

        terrain = getattr(self._env.scene, "terrain", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        max_terrain_level = float(getattr(terrain, "max_terrain_level", 1) or 1)
        if terrain_levels is not None and max_terrain_level > 1.0:
            level_ratio = terrain_levels[env_ids].float() / max(max_terrain_level - 1.0, 1.0)
            min_scale = float(self.target_cfg.terrain_speed_min_scale)
            scale = 1.0 - (1.0 - min_scale) * torch.clamp(level_ratio, 0.0, 1.0)
            vx_max = vx_max * scale

        stairs_mask, _, _ = self._stair_masks(env_ids)
        if torch.any(stairs_mask):
            stair_limit = torch.full_like(vx_max, float(self.target_cfg.stair_vx_max))
            vx_max = torch.where(stairs_mask, torch.minimum(vx_max, stair_limit), vx_max)
        return vx_max

    def _set_velocity_command(self, env_ids: Sequence[int] | slice | None = None):
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = self._env_ids_to_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        self.vel_command_b[env_ids_tensor] = 0.0
        self.metrics["scheduled_vx_max"][env_ids_tensor] = 0.0

        target_env_ids = env_ids_tensor[self.command_mode[env_ids_tensor] == self.MODE_TARGET]
        if target_env_ids.numel() > 0:
            distance = self.goal_distance[target_env_ids]
            heading_error = self.goal_heading_error[target_env_ids]
            vx_max = self._scheduled_vx_max(target_env_ids)
            distance_scale = torch.clamp(distance / float(self.target_cfg.distance_to_max_speed), 0.0, 1.0)
            turn_gate = torch.clamp(torch.cos(heading_error), 0.0, 1.0)
            yaw_rate = torch.clamp(
                self.cfg.heading_control_stiffness * heading_error,
                min=float(self.cfg.ranges.ang_vel_z[0]),
                max=float(self.cfg.ranges.ang_vel_z[1]),
            )

            self.vel_command_b[target_env_ids, 0] = vx_max * distance_scale * turn_gate
            self.vel_command_b[target_env_ids, 1] = 0.0
            self.vel_command_b[target_env_ids, 2] = yaw_rate
            self.metrics["scheduled_vx_max"][target_env_ids] = vx_max

        direct_env_ids = env_ids_tensor[self.command_mode[env_ids_tensor] != self.MODE_TARGET]
        if direct_env_ids.numel() > 0:
            self.vel_command_b[direct_env_ids] = self.direct_command_b[direct_env_ids]

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
