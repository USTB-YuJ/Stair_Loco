from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from legged_gym.utils.terrain import Terrain


from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.warp_render_v3 import DepthRendererWarp, depth_image_preprocessing
from legged_gym.utils.depth_roi import crop_window_from_pixels
from .legged_robot_config import LeggedRobotCfg
import torch.nn.functional as F
import random
import cv2


def adaptive_gaussian_filter(depth_map, device, kernel_size=5, sigma=1.0):
    normal_part = cv2.GaussianBlur(depth_map, (kernel_size, kernel_size), sigma)
    return torch.from_numpy(normal_part).to(device)

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.target_q_list = []
        self.q_list = []

        self.global_counter = 0
        self.total_env_steps_counter = 0
        
        self.init_done = True

    def render(self, sync_frame_time=True):
        if self.viewer and getattr(self.cfg.terrain, "visualize_safety_map", False):
            self._draw_terrain_safety_overlay()
        return super().render(sync_frame_time=sync_frame_time)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # dr: action delay
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = self.cfg.domain_rand.action_curr_step.pop(0)
            indices = torch.randint(
                low=-self.delay -1,
                high=0,
                size=(self.num_envs, 1),
                device=self.device,
                dtype=torch.long
            )
            actions = self.action_history_buf[indices] # delay for 1/50=20ms

        self.global_counter += 1
        self.total_env_steps_counter += 1

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
        reset_env_ids, terminal_amp_states, terminal_obs, terminal_critic_obs = self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        if (self.cfg.depth.use_camera or self.cfg.depth.warp_camera) and self.global_counter % self.cfg.depth.update_interval == 0:
            if self.cfg.depth.warp_camera:
                self.extras["depth"] = self.warp_depth_buffer
            else:
                self.extras["depth"] = self.depth_buffer
        else:
            self.extras["depth"] = None
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states, terminal_obs, terminal_critic_obs

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image

    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def _crop_and_resize_depth_images(self, depth_images):
        """Apply configured ROI crop and return tensors shaped like cfg.depth.resized."""
        if self.cfg.depth.crop_depth:
            crop_top, crop_bottom, crop_left, crop_right = crop_window_from_pixels(
                depth_images.shape[-2:], self.cfg.depth.crop_pixels
            )
            depth_images = depth_images[..., crop_top:crop_bottom, crop_left:crop_right]

        target_size = tuple(self.cfg.depth.resized)
        if tuple(depth_images.shape[-2:]) == target_size:
            return depth_images.contiguous()

        if depth_images.dim() == 2:
            return F.interpolate(
                depth_images.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).squeeze(0)
        if depth_images.dim() == 3:
            return F.interpolate(
                depth_images.unsqueeze(1),
                size=target_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        raise ValueError(f"Unsupported depth tensor shape: {tuple(depth_images.shape)}")

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return

        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
                                                                self.envs[i],
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)

            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            depth_image = self._crop_and_resize_depth_images(depth_image)

            init_flag = self.episode_length_buf <= 1

            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
                                                dim=0)
        self.gym.end_access_image_tensors(self.sim)
    
    def warp_update_depth_buffer(self):
        if not self.cfg.depth.warp_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return

        base_camera_quat = self.rigid_body_states.view(self.num_envs, self.num_bodies, -1)[:, 0, 3:7]
        base_pos_w = self.rigid_body_states.view(self.num_envs, self.num_bodies, -1)[:, 0, :3]

        if self.cfg.depth.enable_self_occlusion and hasattr(self.warp_renderer, "update_robot_meshes"):
            rb = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
            self.warp_renderer.update_robot_meshes(rb)

        depth_images = self.warp_renderer.render_depth(
            base_pos=base_pos_w,
            base_quat=base_camera_quat,
        )
        from pytorch3d import transforms as _t3d
        _base_rot = _t3d.quaternion_to_matrix(
            torch.cat([base_camera_quat[:, 3:], base_camera_quat[:, :3]], dim=1))
        self._cam_rot_w = _base_rot
        self._cam_pos_w = base_pos_w
        # store full warp2cam for mask generation
        from legged_gym.sensors.depth_camera import quat_pos_2_mat_torch
        gym2robot = quat_pos_2_mat_torch(base_pos_w, base_camera_quat).to(self.device)
        self._warp2cam = self.warp_renderer.warp2gym.unsqueeze(0).repeat([base_pos_w.shape[0],1,1]) @ gym2robot @ self.warp_renderer.robot2cam

        self._depth_stage_raw = depth_images.clone()

        # generate GT safety heatmap from depth via precomputed safety map
        depth_safety = depth_images.clone()
        depth_safety = depth_image_preprocessing(depth_safety, near_plane=self.cfg.depth.near_clip, far_plane=self.cfg.depth.far_clip, depth_scale=1.)
        depth_safety_m = depth_safety  # meters

        # camera intrinsics (reuse existing logic)
        fovy_offset = float(self.warp_renderer.fovy_dist_offset[0].item())
        tan_half = 1.0 / (fovy_offset + 1.0)
        H_orig, W_orig = depth_safety.shape[1], depth_safety.shape[2]
        fy = (H_orig / 2.0) / tan_half
        fx = fy
        cx = W_orig / 2.0
        cy = H_orig / 2.0

        v, u = torch.meshgrid(torch.arange(H_orig, device=self.device, dtype=torch.float32),
                              torch.arange(W_orig, device=self.device, dtype=torch.float32), indexing='ij')
        z = depth_safety_m
        x_c = (u - cx) / fx * z
        y_c = (v - cy) / fy * z

        warp2gym_R = self.warp_renderer.warp2gym[:3, :3]
        gym_from_warp_R = warp2gym_R.T
        cam_pos_warp = self._warp2cam[:, :3, 3]
        cam_pos_world = (gym_from_warp_R @ cam_pos_warp.T).T
        cam_rot_warp = self._warp2cam[:, :3, :3]
        cam_rot_world = gym_from_warp_R.unsqueeze(0) @ cam_rot_warp

        pts_cam_warp = torch.stack([z, -x_c, -y_c], dim=-1)
        pts_world = (cam_rot_world[:, None, None] @ pts_cam_warp.unsqueeze(-1)).squeeze(-1) + cam_pos_world[:, None, None]

        hs = self.terrain.cfg.horizontal_scale
        border = self.terrain.cfg.border_size
        px = ((pts_world[..., 0] + border) / hs).long().clamp(0, self.terrain_safety_map.shape[1] - 1)
        py = ((pts_world[..., 1] + border) / hs).long().clamp(0, self.terrain_safety_map.shape[0] - 1)

        safety_heatmap = self.terrain_safety_map[px, py]  # [B, H_orig, W_orig] match [col,row] convention
        # height_diff filter: exclude body pixels and vertical surfaces
        pts_xy_flat = pts_world[..., :2].reshape(-1, 2)
        terrain_h = self._query_height_at_points(pts_xy_flat).reshape(pts_world.shape[0], pts_world.shape[1], pts_world.shape[2])
        height_diff = torch.abs(pts_world[..., 2] - terrain_h)
        valid = (depth_safety_m > 0) & (height_diff < 0.15)
        safety_heatmap = safety_heatmap * valid.float()


        # add noise to raw depth images
        if self.cfg.depth.gaussian_noise:
            depth_images += torch.randn((self.num_envs, *self.cfg.depth.original), device=self.device) * self.cfg.depth.gaussian_noise_std
        self._depth_stage_gaussian = depth_images.clone()

        if self.cfg.depth.dis_noise != 0:
            depth_images += self.depth_dis_noise[..., None]
        self._depth_stage_dis = depth_images.clone()

        if self.cfg.depth.edge_invalid_noise:
            w = self.cfg.depth.edge_invalid_width
            depth_images[:, :w, :] = 0.
            depth_images[:, -w:, :] = 0.
            depth_images[:, :, :w] = 0.
            depth_images[:, :, -w:] = 0.
        self._depth_stage_edge = depth_images.clone()

        if self.cfg.depth.random_invalid_patch:
            H, W = depth_images.shape[-2], depth_images.shape[-1]
            for _ in range(self.cfg.depth.random_invalid_patch_num):
                ph = torch.randint(1, self.cfg.depth.random_invalid_patch_size + 1, (1,)).item()
                pw = torch.randint(1, self.cfg.depth.random_invalid_patch_size + 1, (1,)).item()
                py = torch.randint(0, H - ph + 1, (1,)).item()
                px = torch.randint(0, W - pw + 1, (1,)).item()
                depth_images[:, py:py+ph, px:px+pw] = 0.
        self._depth_stage_patch = depth_images.clone()

        if self.cfg.depth.depth_discontinuity_noise:
            # invalidate pixels near depth discontinuities (flying pixel artifact)
            d = depth_images  # [B, H, W]
            diff_h = torch.abs(d[:, 1:, :] - d[:, :-1, :])  # [B, H-1, W]
            diff_w = torch.abs(d[:, :, 1:] - d[:, :, :-1])  # [B, H, W-1]
            thresh = self.cfg.depth.depth_discontinuity_thresh
            edge_h = torch.zeros_like(d, dtype=torch.bool)
            edge_w = torch.zeros_like(d, dtype=torch.bool)
            edge_h[:, 1:, :] |= diff_h > thresh
            edge_h[:, :-1, :] |= diff_h > thresh
            edge_w[:, :, 1:] |= diff_w > thresh
            edge_w[:, :, :-1] |= diff_w > thresh
            edge_mask = edge_h | edge_w
            r = self.cfg.depth.depth_discontinuity_dilate
            if r > 0:
                import torch.nn.functional as _F
                k = 2 * r + 1
                edge_mask = _F.max_pool2d(
                    edge_mask.float().unsqueeze(1), kernel_size=k, stride=1, padding=r
                ).squeeze(1).bool()
            depth_images[edge_mask] = 0.
        self._depth_stage_discontinuity = depth_images.clone()

        # clip & normalize depth images
        depth_images = depth_image_preprocessing(depth_images, near_plane=self.cfg.depth.near_clip, far_plane=self.cfg.depth.far_clip, depth_scale=1.)
        depth_images = (depth_images - self.cfg.depth.near_clip) / (
            self.cfg.depth.far_clip - self.cfg.depth.near_clip
        ) - 0.5
        self._depth_stage_normalized = depth_images.clone()

        # gaussian filter
        if self.cfg.depth.gaussian_filter:
            kernel = random.choice(self.cfg.depth.gaussian_filter_kernel)
            sigma = float(np.random.rand(1) * self.cfg.depth.gaussian_filter_sigma)
            depth_images = adaptive_gaussian_filter(depth_images.cpu().numpy(), self.device, kernel, sigma)

        depth_norm_full = depth_images

        # save pre-crop depth for visualization
        self._raw_warp_depth = depth_norm_full.clone()
        depth_images = self._crop_and_resize_depth_images(depth_norm_full)

        init_flag = self.episode_length_buf <= 1

        self.warp_depth_buffer = torch.cat([self.warp_depth_buffer[:, 1:], depth_images.unsqueeze(1)], dim=1)
        self.warp_depth_buffer[init_flag] = torch.stack([depth_images] * self.cfg.depth.buffer_len, dim=1)[init_flag]

        # safety heatmap already computed from clean depth above (before noise)
        safety_heatmap_resized = self._crop_and_resize_depth_images(safety_heatmap)
        self.warp_safety_heatmap_buffer = torch.cat([self.warp_safety_heatmap_buffer[:, 1:], safety_heatmap_resized.unsqueeze(1)], dim=1)
        self.warp_safety_heatmap_buffer[init_flag] = torch.stack([safety_heatmap_resized] * self.cfg.depth.buffer_len, dim=1)[init_flag]

        # body mask: 1=terrain pixel, 0=body pixel (excluded from seg_loss)
        body_mask = torch.ones_like(depth_images)  # all terrain for now (body detect via depth discontinuity TBD)  # body is closer than terrain
        body_mask = body_mask.float()
        body_mask_resized = self._crop_and_resize_depth_images(body_mask)
        self.warp_body_mask_buffer = torch.cat([self.warp_body_mask_buffer[:, 1:], body_mask_resized.unsqueeze(1)], dim=1)
        self.warp_body_mask_buffer[init_flag] = torch.stack([body_mask_resized] * self.cfg.depth.buffer_len, dim=1)[init_flag]


    def _draw_terrain_safety_overlay(self):
        """Draw terrain safety map on the mesh surface (green safe, red danger)."""
        if self.viewer is None:
            return
        if not hasattr(self, "terrain_safety_map"):
            return
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            return
        if not hasattr(self, "terrain_levels") or not hasattr(self, "terrain_types"):
            return

        env_id = int(self.lookat_id) if hasattr(self, "lookat_id") else 0
        row = int(self.terrain_levels[env_id].item())
        col = int(self.terrain_types[env_id].item())

        border_px = self.terrain.border
        len_px = self.terrain.length_per_env_pixels
        wid_px = self.terrain.width_per_env_pixels
        start_x = border_px + row * len_px
        end_x = start_x + len_px
        start_y = border_px + col * wid_px
        end_y = start_y + wid_px

        spacing = getattr(self.cfg.terrain, "safety_map_sample_spacing", 0.2)
        stride = max(1, int(round(spacing / self.terrain.cfg.horizontal_scale)))
        n_x = max(1, (end_x - start_x + stride - 1) // stride)
        n_y = max(1, (end_y - start_y + stride - 1) // stride)
        max_points = 2500
        if n_x * n_y > max_points:
            factor = int(np.ceil(np.sqrt((n_x * n_y) / max_points)))
            stride *= max(1, factor)

        xs_idx = torch.arange(start_x, end_x, stride, device=self.device)
        ys_idx = torch.arange(start_y, end_y, stride, device=self.device)
        if xs_idx.numel() == 0 or ys_idx.numel() == 0:
            return

        safety = self.terrain_safety_map[start_x:end_x:stride, start_y:end_y:stride]
        height = self.height_samples[start_x:end_x:stride, start_y:end_y:stride].float() * self.terrain.cfg.vertical_scale

        scale = self.terrain.cfg.horizontal_scale
        border_m = self.terrain.cfg.border_size
        xs = xs_idx.float() * scale - border_m
        ys = ys_idx.float() * scale - border_m
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        z = height

        pts = torch.stack([grid_x, grid_y, z], dim=-1).reshape(-1, 3).cpu().numpy()
        safety_vals = safety.reshape(-1).clamp(0.0, 1.0).cpu().numpy()

        if not hasattr(self, "_safety_viz_geoms"):
            radius = max(0.02, 0.25 * scale)
            colors = [(1.0, 0.0, 0.0), (0.75, 0.25, 0.0), (0.5, 0.5, 0.0), (0.25, 0.75, 0.0), (0.0, 1.0, 0.0)]
            self._safety_viz_geoms = [gymutil.WireframeSphereGeometry(radius, 6, 6, None, color=c) for c in colors]

        self.gym.clear_lines(self.viewer)
        bins = len(self._safety_viz_geoms)
        z_offset = 0.02
        for point, s in zip(pts, safety_vals):
            idx = min(bins - 1, int(s * bins))
            pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2] + z_offset), r=None)
            gymutil.draw_lines(self._safety_viz_geoms[idx], self.gym, self.viewer, self.envs[env_id], pose)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_base_lin_vel[:] = self.base_lin_vel[:]


    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1000., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        if self.cfg.env.test:
            # y termination
            if self.cfg.terrain.mesh_type == "trimesh":
                offset_y = torch.abs(self.root_states[:, 1] - self.origin_y)
                only_forward_env = (self.env_class!=0)
                self.reset_buf |= torch.logical_and(only_forward_env, offset_y>1.0)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum and self.cfg.terrain.mesh_type == "trimesh":
            self._update_terrain_curriculum(env_ids)
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        if self.cfg.depth.use_camera:
            self.depth_buffer[env_ids] = 0.
        if self.cfg.depth.warp_camera:
            self.warp_depth_buffer[env_ids] = 0.

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s 
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / ((self.episode_length_buf[env_ids]+1) * self.dt))
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
       
        # send timeout info to the algorithm
        # log additional curriculum info
        if self.cfg.terrain.curriculum and self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self.episode_length_buf[env_ids] = 0

        self.depth_dis_noise[env_ids] = torch_rand_float(-self.cfg.depth.dis_noise, self.cfg.depth.dis_noise, (len(env_ids), 1), device=self.device)

        self.update_dr_params(env_ids)

    def update_dr_params(self, env_ids):
        self.push_interval = torch_rand_float(self.cfg.domain_rand.push_interval_min, self.cfg.domain_rand.push_interval_max, (self.num_envs,1), device=self.device).round().long().squeeze(dim=-1)

        # reset randomized prop
        if self.cfg.domain_rand.randomize_gains:
            new_randomized_gains = self.compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), self.num_dof), device=self.device)
        
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        print("*"*80)
        print("Start creating ground...")
        start = time.time()
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time.time() - start))
        print("*"*80)
        # self._create_ground_plane()
        if self.cfg.depth.warp_camera and mesh_type in ['heightfield', 'trimesh']:

            pitch = torch_rand_float(self.cfg.depth.y_angle[0], self.cfg.depth.y_angle[1], (self.num_envs, 1), device=self.device)
            yaw = torch_rand_float(self.cfg.depth.z_angle[0], self.cfg.depth.z_angle[1], (self.num_envs, 1), device=self.device)
            roll = torch_rand_float(self.cfg.depth.x_angle[0], self.cfg.depth.x_angle[1], (self.num_envs, 1), device=self.device)
            euler = torch.deg2rad(torch.cat([roll, pitch, yaw], dim=-1))
        
            camera_pos = torch.tensor(self.cfg.depth.position, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            if self.cfg.depth.rand_position:
                camera_pos_x = torch_rand_float(self.cfg.depth.x_pos_range[0], self.cfg.depth.x_pos_range[1], (self.num_envs, 1), device=self.device)
                camera_pos_y = torch_rand_float(self.cfg.depth.y_pos_range[0], self.cfg.depth.y_pos_range[1], (self.num_envs, 1), device=self.device)
                camera_pos_z = torch_rand_float(self.cfg.depth.z_pos_range[0], self.cfg.depth.z_pos_range[1], (self.num_envs, 1), device=self.device)
                camera_pos_offset = torch.cat([camera_pos_x, camera_pos_y, camera_pos_z], dim=-1)
                camera_pos += camera_pos_offset

            if self.cfg.depth.use_camera:
                self.camera_pos = camera_pos
                self.camera_y_angle = pitch
                self.camera_z_angle = yaw
                self.camera_x_angle = roll

            camera_fovy = torch_rand_float(self.cfg.depth.fovy_range[0], self.cfg.depth.fovy_range[1], (self.num_envs, 1), device=self.device)
            far_t = float(self.cfg.depth.far_clip) + 0.2
            self.warp_renderer = DepthRendererWarp(
                image_params=self.cfg.depth.original,
                cam2base_xyz=camera_pos,
                cam2base_euler=euler,
                fovy=camera_fovy,
                device=self.device,
                num_envs=self.num_envs,
                far_t=far_t,
                miss_t=far_t,
            )
            self.terrain.vertices[:, :2] = self.terrain.vertices[:, :2] - self.cfg.terrain.border_size
            self.warp_renderer.render_mesh(self.terrain.vertices, self.terrain.triangles)
        self._create_envs()
        if self.cfg.depth.enable_self_occlusion:
            self._init_self_occlusion_meshes()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            rng = self.cfg.domain_rand.friction_range
            self.randomized_frictions[env_id] = np.random.uniform(rng[0], rng[1])
            for s in range(len(props)):
                props[s].friction = self.randomized_frictions[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        
        return props

    def refresh_actor_rigid_shape_props(self, env_ids):
        # dr: friction
        if self.cfg.domain_rand.randomize_friction:
            self.randomized_frictions[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        # dr: restitution
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.randomized_frictions[env_id, 0]
                rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # dr: payload mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            added_mass = np.random.uniform(rng[0], rng[1])
            self.randomized_added_masses[env_id] = added_mass
            props[self.torso_body_index].mass += added_mass
        # dr: com position
        if self.cfg.domain_rand.randomize_com_pos:
            rng = self.cfg.domain_rand.com_x_pos_range
            com_x_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,0] = com_x_pos
            rng = self.cfg.domain_rand.com_y_pos_range
            com_y_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,1] = com_y_pos
            rng = self.cfg.domain_rand.com_z_pos_range
            com_z_pos = np.random.uniform(rng[0], rng[1])
            self.randomized_com_pos[env_id,2] = com_z_pos
            props[self.torso_body_index].com += gymapi.Vec3(com_x_pos,com_y_pos,com_z_pos)
        # dr: link mass
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                props[i].mass = props[i].mass * np.random.uniform(rng[0], rng[1])

        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        if control_type=="P":
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains*self.dof_vel
        elif control_type=="V":
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        if(self.cfg.domain_rand.randomize_motor_strength):
            torques = torques * self.motor_strength + self.actuation_offset

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :1] += torch_rand_float(-0.5, 0.2, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
            self.root_states[env_ids, 1:2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
            self.origin_y[env_ids] = self.root_states[env_ids, 1].clone()
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        # push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % self.push_interval == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        min_vel = self.cfg.domain_rand.min_push_vel_xy
        if min_vel < max_vel:
            signal = torch.randint(0, 2, (self.num_envs, 2), device=self.device, dtype=torch.bool)
            pos_push = torch_rand_float(min_vel, max_vel, (self.num_envs, 2), device=self.device)
            neg_push = torch_rand_float(-max_vel, -min_vel, (self.num_envs, 2), device=self.device)
            if self.cfg.domain_rand.stair_no_push:
                self.root_states[self.env_class!=3, 7:9] = torch.where(signal, pos_push, neg_push)[self.env_class!=3] # lin vel x/y
            self.root_states[:, 7:9] = torch.where(signal, pos_push, neg_push)[self.env_class!=3] # lin vel x/y
        else:
            if self.cfg.domain_rand.stair_no_push:
                push_mask = self.env_class!=3
                self.root_states[push_mask, 7:9] = torch_rand_float(-max_vel, max_vel, (push_mask.sum(), 2), device=self.device) # lin vel x/y
            self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        
        dis_to_origin = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s
        move_up = dis_to_origin > 0.8*threshold
        move_down = dis_to_origin < 0.4*threshold

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.knee_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.knee_indices, 0:3]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        self.sensor_forces = force_sensor_readings.view(self.num_envs, 2, 6)[..., :3]

        self.camera_offset = torch.tensor([self.cfg.depth.position[0], self.cfg.depth.position[1], self.cfg.depth.position[2]]
                                          , device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.knee_pos_in_body = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time_record = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        self.footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        self.depth_dis_noise = torch.zeros(self.num_envs, 1, device=self.device)
        self.footdis_record = torch.zeros(self.num_envs, len(self.feet_indices), 2, device=self.device)
        self.first_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.footdis_record_contact = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.footdis_record_leave = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs, device=self.device, dtype=torch.float)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.base_height_points = self._init_base_height_points()
        self.measured_heights = 0
        self.measured_forward_heights = 0
        

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.motor_strength = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains = self.compute_randomized_gains(self.num_envs)

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[0],
                                            self.cfg.depth.resized[1]).to(self.device)
        if self.cfg.depth.warp_camera:
            self.warp_depth_buffer = torch.zeros(self.num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[0],
                                            self.cfg.depth.resized[1]).to(self.device)
            self._raw_warp_depth = None  # pre-crop depth for visualization
            self.warp_safety_heatmap_buffer = torch.zeros(self.num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[0],
                                            self.cfg.depth.resized[1]).to(self.device)
            self.warp_body_mask_buffer = torch.zeros(self.num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[0],
                                            self.cfg.depth.resized[1]).to(self.device)
            self._cam_rot_w = None
            self._cam_pos_w = None
            self._warp2cam = None

    def compute_randomized_gains(self, num_envs):
        p_mult = torch_rand_float(self.cfg.domain_rand.stiffness_multiplier_range[0], self.cfg.domain_rand.stiffness_multiplier_range[1],
                                  (num_envs, self.num_actions), device=self.device)
        d_mult = torch_rand_float(self.cfg.domain_rand.damping_multiplier_range[0], self.cfg.domain_rand.damping_multiplier_range[1],
                                  (num_envs, self.num_actions), device=self.device)
        return p_mult * self.p_gains, d_mult * self.d_gains

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
            Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        print("Trimesh added")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.stuck_mask = torch.tensor(self.terrain.stuck_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.stair_pen_mask = torch.tensor(self.terrain.stair_pen_mask).view(2, self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

        # Precompute flatness map from height field: exp(-local_var / sigma^2)
        h = torch.tensor(self.terrain.height_field_raw, dtype=torch.float32, device=self.device) * self.terrain.cfg.vertical_scale
        h_3d = h.unsqueeze(0).unsqueeze(0)
        local_mean = F.avg_pool2d(h_3d, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((h_3d - local_mean)**2, kernel_size=3, stride=1, padding=1)
        self.flatness_map = torch.exp(-local_var.squeeze() / (0.02**2))
        self.flatness_map = torch.exp(-local_var.squeeze() / (0.02**2))

        # Precompute terrain safety map: nz * 0.5 + flatness * 0.5
        with torch.no_grad():
            vs = self.terrain.cfg.vertical_scale
            hs = self.terrain.cfg.horizontal_scale
            h = torch.tensor(self.terrain.height_field_raw, dtype=torch.float32, device=self.device) * vs
            dz_dx = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2 * hs)
            dz_dy = (h[1:-1, 2:] - h[1:-1, :-2]) / (2 * hs)
            nz = 1.0 / torch.sqrt(1.0 + dz_dx**2 + dz_dy**2)
            nz = torch.nn.functional.pad(nz[None, None], (1, 1, 1, 1), mode='replicate').squeeze()
            self.terrain_safety_map = 0.5 * nz + 0.5 * self.flatness_map
        print(f"[SAFETY_MAP] Precomputed terrain_safety_map shape={self.terrain_safety_map.shape}")

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[1]
            camera_props.height = self.cfg.depth.original[0]
            camera_props.enable_tensors = True
            camera_props.horizontal_fov = self.cfg.depth.horizontal_fov
            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            # set same camera params as warp_camera
            if self.cfg.depth.warp_camera:
                camera_position = self.camera_pos[i]
                camera_y_angle = self.camera_y_angle[i].cpu().numpy()
                camera_z_angle = self.camera_z_angle[i].cpu().numpy()
                camera_x_angle = self.camera_x_angle[i].cpu().numpy()
            else:
                camera_position = np.copy(config.position)
                camera_y_angle = np.random.uniform(config.y_angle[0], config.y_angle[1])
                camera_z_angle = np.random.uniform(config.z_angle[0], config.z_angle[1])
                camera_x_angle = np.random.uniform(config.x_angle[0], config.x_angle[1])
            
            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(np.radians(camera_z_angle),
                                                           np.radians(camera_y_angle), np.radians(camera_x_angle))
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform,
                                           gymapi.FOLLOW_TRANSFORM)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names = body_names
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # use the sensor to acquire contact force, may be more accurate
        sensor_pose = gymapi.Transform()
        for name in feet_names:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False  # for example gravity
            sensor_options.enable_constraint_solver_forces = True  # for example contacts
            sensor_options.use_world_frame = True  # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        torso_candidates = []
        torso_name_cfg = getattr(self.cfg.asset, "torso_name", None)
        if torso_name_cfg:
            torso_candidates.append(torso_name_cfg)
        # common torso/base names across different Unitree-style URDFs
        torso_candidates.extend(["torso_link", "torso", "trunk", "base", "base_link", "pelvis"])
        self.torso_body_index = None
        for torso_name in torso_candidates:
            if torso_name in body_names:
                self.torso_body_index = body_names.index(torso_name)
                break
        if self.torso_body_index is None:
            # fall back to first rigid body to avoid hard crash; mass/COM randomization will be less precise
            self.torso_body_index = 0
            print(
                f"[LeggedRobot] WARNING: could not find torso body from candidates {torso_candidates}. "
                f"Available bodies (first 20): {body_names[:20]}"
            )
        self.randomized_frictions = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.randomized_added_masses = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.randomized_com_pos = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.origin_y = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            self.origin_y[i] = pos[1]
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            if self.cfg.depth.use_camera:
                self.attach_camera(i, env_handle, actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _init_self_occlusion_meshes(self):
        if not self.cfg.depth.enable_self_occlusion:
            return
        if not hasattr(self, "warp_renderer") or self.warp_renderer is None:
            return
        if not hasattr(self.warp_renderer, "init_robot_meshes"):
            return

        module_name = getattr(self.cfg.depth, "robot_geom_module", "")
        if not module_name:
            print("[SELF_OCCLUSION] robot_geom_module not set; skipping self-occlusion BVH.")
            return

        try:
            import importlib
            from legged_gym.utils.robot_geom import build_robot_template
            geom_mod = importlib.import_module(module_name)
        except Exception as exc:
            print(f"[SELF_OCCLUSION] Failed to import {module_name}: {exc}")
            return

        link_geoms = None
        for attr in ("LINK_GEOMS", "H1_LINK_GEOMS", "G1_LINK_GEOMS"):
            if hasattr(geom_mod, attr):
                link_geoms = getattr(geom_mod, attr)
                break
        if link_geoms is None:
            print(f"[SELF_OCCLUSION] No link geometry list found in {module_name}.")
            return

        template = build_robot_template(link_geoms)
        body_indices = []
        missing = []
        for name in template.link_names:
            if name in self.body_names:
                body_indices.append(self.body_names.index(name))
            else:
                missing.append(name)
        if missing:
            print(f"[SELF_OCCLUSION] Missing body names ({len(missing)}): {missing[:5]}")
            return

        self.warp_renderer.init_robot_meshes(
            template_verts_local=template.verts_local,
            template_tris=template.tris,
            vert_to_link=template.vert_to_link,
            body_indices=np.array(body_indices, dtype=np.int32),
            refit_stride=getattr(self.cfg.depth, "refit_stride", 1),
        )
        print(f"[SELF_OCCLUSION] Initialized robot BVHs: verts={template.num_verts} tris={template.num_tris}")

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
     

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval_max = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.push_interval_min = np.ceil(self.cfg.domain_rand.push_interval_min_s / self.dt)

    def _draw_camera(self):
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 10, 10, None, color=(1, 0, 0))

        camera_quat = self.rigid_body_states.view(self.num_envs, self.num_bodies, -1)[:, 0, 3:7]
        camera_local_pos = quat_rotate(camera_quat, self.camera_offset)
        base_positions = self.rigid_body_states.view(self.num_envs, self.num_bodies, -1)[:, 0, :3]
        camera_positions = base_positions + camera_local_pos
        for i, goal in enumerate(camera_positions):
            pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal[2]), r=None)
            gymutil.draw_lines(
                sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose
            )
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _draw_knee(self):
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 10, 10, None, color=(1, 0, 0))
        pos = self.knee_pos[self.lookat_id].cpu().numpy()
        for i, point in enumerate(pos):
            pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)
            gymutil.draw_lines(
                sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose
            )
    
    def _compute_edge_score(self, depth_images, sigma_edge=0.1):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=depth_images.dtype, device=depth_images.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=depth_images.dtype, device=depth_images.device).view(1, 1, 3, 3)
        grad_x = F.conv2d(depth_images.unsqueeze(1), sobel_x, padding=1)
        grad_y = F.conv2d(depth_images.unsqueeze(1), sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return torch.exp(-grad_mag**2 / sigma_edge**2).squeeze(1)

    def _generate_safety_heatmap(self, depth_images):
        """Binary stair-tread mask via backprojection + terrain height query.
        depth_images: [B, H, W] normalized depth at original resolution (before crop).
        """

        if self.height_samples is None or self._warp2cam is None:
            return torch.zeros_like(depth_images)
        B, H, W = depth_images.shape
        # denormalize to meters
        depth_m = (depth_images + 0.5) * (self.cfg.depth.far_clip - self.cfg.depth.near_clip) + self.cfg.depth.near_clip

        # camera intrinsics from fovy (original resolution H x W)
        fovy_offset = float(self.warp_renderer.fovy_dist_offset[0].item())
        # fovy_dist_offset = 1/tan(fovy/2) - 1  =>  tan(fovy/2) = 1/(fovy_offset+1)
        tan_half = 1.0 / (fovy_offset + 1.0)
        # fy based on height (warp uses height as the reference axis)
        fy = (H / 2.0) / tan_half
        fx = fy
        cx = W / 2.0
        cy = H / 2.0

        v, u = torch.meshgrid(torch.arange(H, device=self.device, dtype=torch.float32),
                              torch.arange(W, device=self.device, dtype=torch.float32), indexing='ij')
        z = depth_m  # [B, H, W]
        x_c = (u - cx) / fx * z
        y_c = (v - cy) / fy * z
        # pts in camera frame (x-right, y-down, z-forward)
        pts_cam = torch.stack([x_c, y_c, z], dim=-1)  # [B, H, W, 3]

        # camera world pose: _warp2cam is cam->warp; warp2gym is gym->warp
        # use inverse rotation to go warp->gym
        warp2gym_R = self.warp_renderer.warp2gym[:3, :3]  # [3,3]
        gym_from_warp_R = warp2gym_R.T
        # camera position in gym world frame
        cam_pos_warp = self._warp2cam[:, :3, 3]  # [B, 3]
        cam_pos_world = (gym_from_warp_R @ cam_pos_warp.T).T  # [B, 3]
        # camera rotation in gym world frame: R_world = R_gym_from_warp @ R_warp
        cam_rot_warp = self._warp2cam[:, :3, :3]  # [B, 3, 3]
        cam_rot_world = gym_from_warp_R.unsqueeze(0) @ cam_rot_warp  # [B, 3, 3]

        # backproject: pts_world = R_cam_world @ pts_cam + cam_pos_world
        # Note: warp camera frame is (x-forward, y-left, z-up) after warp2gym
        # pts_cam here is in standard pinhole (x-right, y-down, z-forward)
        # convert to warp camera frame: x_warp=z, y_warp=-x, z_warp=-y
        pts_cam_warp = torch.stack([z, -x_c, -y_c], dim=-1)  # [B, H, W, 3]
        pts_world = (cam_rot_world[:, None, None] @ pts_cam_warp.unsqueeze(-1)).squeeze(-1) \
                    + cam_pos_world[:, None, None]

        pts_xy = pts_world[..., :2].reshape(B * H * W, 2)
        terrain_h = self._query_height_at_points(pts_xy).reshape(B, H, W)
        terrain_nz = self._query_normal_z_at_points(pts_xy).reshape(B, H, W)
        flatness = self._query_flatness_at_points(pts_xy).reshape(B, H, W)
        height_diff = torch.abs(pts_world[..., 2] - terrain_h)

        # edge score from depth image gradient
        edge_score = self._compute_edge_score(depth_m)

        # fuse 3 components into continuous safety heatmap
        w1, w2, w3 = 0.4, 0.3, 0.3
        safety = w1 * terrain_nz + w2 * flatness + w3 * edge_score

        valid = (depth_m > 0) & (height_diff < 0.2)
        safety = safety * valid.float()

        return safety

    def _get_foot_safety(self, foot_xy):
        """foot_xy: [N, 2] world (x,y) -> safety [N] from precomputed map."""
        s = self.terrain.cfg.horizontal_scale
        b = self.terrain.cfg.border_size
        px = ((foot_xy[:, 0] + b) / s).long().clamp(0, self.terrain_safety_map.shape[1] - 1)
        py = ((foot_xy[:, 1] + b) / s).long().clamp(0, self.terrain_safety_map.shape[0] - 1)
        return self.terrain_safety_map[px, py]

    def _query_height_at_points(self, points_xy):
        px = (points_xy[:, 0] + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale
        py = (points_xy[:, 1] + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale
        px = torch.clip(px.long(), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py.long(), 0, self.height_samples.shape[1] - 2)
        h = (self.height_samples[px, py] + self.height_samples[px+1, py] +
             self.height_samples[px, py+1] + self.height_samples[px+1, py+1]) / 4.0
        return h * self.terrain.cfg.vertical_scale

    def _query_normal_z_at_points(self, points_xy):
        s = self.terrain.cfg.horizontal_scale
        px = (points_xy[:, 0] + self.terrain.cfg.border_size) / s
        py = (points_xy[:, 1] + self.terrain.cfg.border_size) / s
        px = torch.clip(px.long(), 1, self.height_samples.shape[0] - 2)
        py = torch.clip(py.long(), 1, self.height_samples.shape[1] - 2)
        vs = self.terrain.cfg.vertical_scale
        dz_dx = (self.height_samples[px+1, py] - self.height_samples[px-1, py]).float() * vs / (2 * s)
        dz_dy = (self.height_samples[px, py+1] - self.height_samples[px, py-1]).float() * vs / (2 * s)
        return 1.0 / torch.sqrt(1.0 + dz_dx**2 + dz_dy**2)

    def _query_flatness_at_points(self, points_xy):
        s = self.terrain.cfg.horizontal_scale
        px = (points_xy[:, 0] + self.terrain.cfg.border_size) / s
        py = (points_xy[:, 1] + self.terrain.cfg.border_size) / s
        px = torch.clip(px.long(), 0, self.flatness_map.shape[0] - 2)
        py = torch.clip(py.long(), 0, self.flatness_map.shape[1] - 2)
        return self.flatness_map[px, py]

    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * self.first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~self.contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_base_z_vel_rate(self):
        return F.relu(torch.abs(self.last_base_lin_vel[:, 2] - self.base_lin_vel[:, 2]) - self.cfg.rewards.max_z_vel_rate)