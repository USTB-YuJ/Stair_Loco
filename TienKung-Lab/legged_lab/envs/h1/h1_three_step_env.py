"""H1 DWAQ environment for the fixed three-step corridor task."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg

from legged_lab.envs.g1.g1_dwaq_env import G1DwaqEnv


MODE_STAIRS = 0
MODE_FLAT_FORWARD = 1
MODE_FLAT_TURN = 2
MODE_FLAT_STAND = 3


def _terrain_columns(terrain_cfg) -> tuple[list[str], dict[str, list[int]]]:
    """Reproduce Isaac Lab's deterministic curriculum column allocation."""
    sub_terrains = terrain_cfg.sub_terrains
    names = list(sub_terrains.keys())
    proportions = np.asarray([sub_terrains[name].proportion for name in names], dtype=np.float64)
    cumulative = np.cumsum(proportions / proportions.sum())
    names_by_column: list[str] = []
    columns_by_name: dict[str, list[int]] = {}

    for column in range(int(terrain_cfg.num_cols)):
        candidates = np.where(column / terrain_cfg.num_cols + 0.001 < cumulative)[0]
        terrain_index = int(candidates[0]) if len(candidates) else len(names) - 1
        name = names[terrain_index]
        names_by_column.append(name)
        columns_by_name.setdefault(name, []).append(column)
    return names_by_column, columns_by_name


class ThreeStepVelocityCommand(UniformVelocityCommand):
    """Sample one episode-long command selected by the terrain task mode."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.task_cfg = env.cfg.three_step_task
        self.terrain_names_by_column, self.terrain_columns = _terrain_columns(
            env.cfg.scene.terrain_generator
        )
        modes = []
        for name in self.terrain_names_by_column:
            if name.startswith("stairs_h"):
                modes.append(MODE_STAIRS)
            elif name == "flat_forward":
                modes.append(MODE_FLAT_FORWARD)
            elif name == "flat_turn":
                modes.append(MODE_FLAT_TURN)
            elif name == "flat_stand":
                modes.append(MODE_FLAT_STAND)
            else:
                raise ValueError(f"Unknown three-step terrain task: {name}")
        self.mode_by_column = torch.tensor(modes, device=self.device, dtype=torch.long)

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.vel_command_b[env_ids] = 0.0
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False

        terrain_types = self._env.scene.terrain.terrain_types[env_ids]
        modes = self.mode_by_column[terrain_types]
        forward_mask = (modes == MODE_STAIRS) | (modes == MODE_FLAT_FORWARD)
        turn_mask = modes == MODE_FLAT_TURN

        forward_ids = env_ids[forward_mask]
        if len(forward_ids) > 0:
            self.vel_command_b[forward_ids, 0].uniform_(*self.task_cfg.forward_velocity_range)

        turn_ids = env_ids[turn_mask]
        if len(turn_ids) > 0:
            magnitudes = torch.empty(len(turn_ids), device=self.device).uniform_(
                *self.task_cfg.turn_yaw_abs_range
            )
            signs = torch.where(
                torch.rand(len(turn_ids), device=self.device) < 0.5,
                -torch.ones_like(magnitudes),
                torch.ones_like(magnitudes),
            )
            self.vel_command_b[turn_ids, 2] = magnitudes * signs

    def _update_command(self):
        # Commands are terrain-bound and remain fixed for the full episode.
        pass


class H1ThreeStepDwaqEnv(G1DwaqEnv):
    """Specialized task environment without changing the standard H1 DWAQ path."""

    def _create_command_generator(self):
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            heading_control_stiffness=0.0,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        return ThreeStepVelocityCommand(cfg=command_cfg, env=self)

    def init_buffers(self):
        super().init_buffers()
        terrain_types = self.scene.terrain.terrain_types
        self.three_step_mode = self.command_generator.mode_by_column[terrain_types]
        self.three_step_forward_mask = (self.three_step_mode == MODE_STAIRS) | (
            self.three_step_mode == MODE_FLAT_FORWARD
        )

        self.three_step_success_x = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        for name, columns in self.command_generator.terrain_columns.items():
            mask = self._column_mask(terrain_types, columns)
            if name.startswith("stairs_h"):
                tread_depth = int(name.rsplit("_d", maxsplit=1)[1]) / 100.0
                self.three_step_success_x[mask] = (
                    self.cfg.three_step_task.approach_distance + 3.0 * tread_depth + 0.10
                )
            elif name == "flat_forward":
                self.three_step_success_x[mask] = self.cfg.three_step_task.flat_forward_success_distance

        self.three_step_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.three_step_fall_buf = torch.zeros_like(self.three_step_success_buf)
        self.three_step_lateral_exit_buf = torch.zeros_like(self.three_step_success_buf)
        self.three_step_backward_exit_buf = torch.zeros_like(self.three_step_success_buf)
        self.three_step_forward_progress_vel = torch.zeros(self.num_envs, device=self.device)
        self.three_step_vel_xy_error_sum = torch.zeros(self.num_envs, device=self.device)
        self.three_step_yaw_error_sum = torch.zeros(self.num_envs, device=self.device)
        self.three_step_prev_local_x = self._local_root_position()[:, 0].clone()

    @staticmethod
    def _column_mask(terrain_types: torch.Tensor, columns: list[int]) -> torch.Tensor:
        mask = torch.zeros_like(terrain_types, dtype=torch.bool)
        for column in columns:
            mask |= terrain_types == column
        return mask

    def _local_root_position(self) -> torch.Tensor:
        return self.robot.data.root_pos_w - self.scene.env_origins

    def update_terrain_levels(self, env_ids):
        """Keep the single terrain row fixed; it is a task mix, not a curriculum."""
        del env_ids
        return {"Terrain/three_step_level": torch.zeros((), device=self.device)}

    def check_reset(self):
        local_root = self._local_root_position()
        delta_x = local_root[:, 0] - self.three_step_prev_local_x
        self.three_step_forward_progress_vel = torch.clamp(
            delta_x / self.step_dt,
            min=0.0,
            max=self.cfg.three_step_task.max_forward_progress_velocity,
        )
        self.three_step_prev_local_x.copy_(local_root[:, 0])

        command = self.command_generator.command
        self.three_step_vel_xy_error_sum += torch.linalg.norm(
            command[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1
        )
        self.three_step_yaw_error_sum += torch.abs(
            command[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
        )

        base_reset, base_timeout = super().check_reset()
        base_failure = base_reset & ~base_timeout
        success = (
            self.three_step_forward_mask
            & (local_root[:, 0] > self.three_step_success_x)
            & ~base_failure
        )
        lateral_exit = (
            torch.abs(local_root[:, 1]) > self.cfg.three_step_task.lateral_exit_distance
        ) & ~success
        backward_exit = (
            self.three_step_forward_mask
            & (local_root[:, 0] < -self.cfg.three_step_task.backward_exit_distance)
            & ~success
        )

        self.three_step_success_buf.copy_(success)
        self.three_step_fall_buf.copy_(base_failure & ~success)
        self.three_step_lateral_exit_buf.copy_(lateral_exit)
        self.three_step_backward_exit_buf.copy_(backward_exit)

        time_out_buf = base_timeout & ~success
        reset_buf = base_reset | success | lateral_exit | backward_exit
        return reset_buf, time_out_buf

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if torch.any(mask):
            return values[mask].float().mean()
        return torch.zeros((), device=values.device)

    def _episode_task_stats(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.sim_step_counter <= 0 or len(env_ids) == 0:
            return {}

        success = self.three_step_success_buf[env_ids]
        forward = self.three_step_forward_mask[env_ids]
        lengths = self.episode_length_buf[env_ids].clamp(min=1)
        elapsed = lengths.float() * self.step_dt
        stats = {
            "Task/success_rate": self._masked_mean(success, forward),
            "Task/time_to_success": self._masked_mean(elapsed, success),
            "Task/timeout_rate": self.time_out_buf[env_ids].float().mean(),
            "Task/fall_rate": self.three_step_fall_buf[env_ids].float().mean(),
            "Task/lateral_exit_rate": self.three_step_lateral_exit_buf[env_ids].float().mean(),
            "Task/backward_exit_rate": self.three_step_backward_exit_buf[env_ids].float().mean(),
        }

        mode_names = {
            MODE_STAIRS: "stairs",
            MODE_FLAT_FORWARD: "flat_forward",
            MODE_FLAT_TURN: "flat_turn",
            MODE_FLAT_STAND: "flat_stand",
        }
        all_modes = self.three_step_mode
        episode_modes = all_modes[env_ids]
        for mode, mode_name in mode_names.items():
            population_mask = all_modes == mode
            episode_mask = episode_modes == mode
            stats[f"Task/Mode/{mode_name}/env_fraction"] = population_mask.float().mean()
            stats[f"Task/Mode/{mode_name}/episode_length"] = self._masked_mean(elapsed, episode_mask)
            xy_error = self.three_step_vel_xy_error_sum[env_ids] / lengths
            yaw_error = self.three_step_yaw_error_sum[env_ids] / lengths
            stats[f"Task/Mode/{mode_name}/vel_xy_error"] = self._masked_mean(xy_error, episode_mask)
            stats[f"Task/Mode/{mode_name}/yaw_error"] = self._masked_mean(yaw_error, episode_mask)

        episode_terrain_types = self.scene.terrain.terrain_types[env_ids]
        stair_names = [
            name for name in self.command_generator.terrain_columns if name.startswith("stairs_h")
        ]
        for name in stair_names:
            mask = self._column_mask(
                episode_terrain_types, self.command_generator.terrain_columns[name]
            )
            if torch.any(mask):
                prefix = f"Task/Stairs/{name.removeprefix('stairs_')}"
                stats[f"{prefix}/success_rate"] = success[mask].float().mean()
                stats[f"{prefix}/time_to_success"] = self._masked_mean(elapsed[mask], success[mask])

        for height_cm in (14, 15, 16, 17):
            names = [name for name in stair_names if name.startswith(f"stairs_h{height_cm}_")]
            mask = torch.zeros_like(episode_terrain_types, dtype=torch.bool)
            for name in names:
                mask |= self._column_mask(
                    episode_terrain_types, self.command_generator.terrain_columns[name]
                )
            if torch.any(mask):
                stats[f"Task/Stairs/height_{height_cm}cm/success_rate"] = success[mask].float().mean()

        for depth_cm in (28, 30, 32, 34, 36):
            names = [name for name in stair_names if name.endswith(f"_d{depth_cm}")]
            mask = torch.zeros_like(episode_terrain_types, dtype=torch.bool)
            for name in names:
                mask |= self._column_mask(
                    episode_terrain_types, self.command_generator.terrain_columns[name]
                )
            if torch.any(mask):
                stats[f"Task/Stairs/depth_{depth_cm}cm/success_rate"] = success[mask].float().mean()
        return stats

    def reset(self, env_ids):
        stats = self._episode_task_stats(env_ids)
        super().reset(env_ids)
        if len(env_ids) == 0:
            return

        local_root = self._local_root_position()
        self.three_step_prev_local_x[env_ids] = local_root[env_ids, 0]
        self.three_step_success_buf[env_ids] = False
        self.three_step_fall_buf[env_ids] = False
        self.three_step_lateral_exit_buf[env_ids] = False
        self.three_step_backward_exit_buf[env_ids] = False
        self.three_step_forward_progress_vel[env_ids] = 0.0
        self.three_step_vel_xy_error_sum[env_ids] = 0.0
        self.three_step_yaw_error_sum[env_ids] = 0.0
        self.extras["log"].update(stats)

