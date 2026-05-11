"""MuJoCo deployment for the H1 loco-with-depth policy.

Loads `policy_depth_1.pt` (the JIT module exported by `PolicyExporterDepth`,
see `legged_gym/utils/helpers.py`) and rolls it out in MuJoCo.

The exported policy expects exactly three inputs:

    actions = policy(
        obs:     Tensor[1, 39],          # current proprio observation
        history: Tensor[1, 10, 39],      # last 10 proprio observations
        depth:   Tensor[1, D, 64, 64],   # D-frame depth ROI stack (H, W)
    ) -> Tensor[1, 10]                   # action mean for the 10 leg joints

    where D = depth_buffer_len from the YAML config.

The 39-dim observation layout matches `H1_Loco_Robot.compute_observations`:
    [0:3]   commands        * cmd_scale       ([vx, vy, wz])
    [3:6]   base_ang_vel    * ang_vel_scale
    [6:9]   projected_gravity
    [9:19]  (dof_pos - default) * dof_pos_scale
    [19:29] dof_vel         * dof_vel_scale
    [29:39] last actions

Depth preprocessing (matches `legged_robot.warp_update_depth_buffer`):
    1. Render depth in meters from the pelvis-mounted camera.
    2. Resize the full frame to 96x128 and crop [32,32,32,0] to 64x64.
    3. Clip to [near_clip, far_clip] = [0, 2] m.
    4. Normalize to [-0.5, 0.5] via `(d - near) / (far - near) - 0.5`.
    5. Push to a length depth_buffer_len ring buffer and feed the full
       stack to the policy, matching `play.py` at runtime.
"""

import argparse
import math
import os
import time
from collections import deque

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from legged_gym import LEGGED_GYM_ROOT_DIR


def get_gravity_orientation(quaternion):
    """Convert a wxyz quaternion (base->world) to projected gravity in body frame."""
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    g = np.zeros(3, dtype=np.float32)
    g[0] = 2.0 * (-qz * qx + qw * qy)
    g[1] = -2.0 * (qz * qy + qw * qx)
    g[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return g


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def normalize_depth(depth_image, near_clip, far_clip):
    """Match `warp_update_depth_buffer` preprocessing in the training env."""
    depth = np.clip(depth_image, near_clip, far_clip)
    depth = (depth - near_clip) / (far_clip - near_clip) - 0.5
    return depth.astype(np.float32)


def make_depth_scene_option(hide_robot: bool, terrain_group: int = 2):
    """Build a scene option that (optionally) hides robot geoms.

    The training warp renderer only renders the static terrain mesh and skips
    the robot's own body, so the policy never saw self-occlusion during the
    first stage. To mimic that distribution in MuJoCo, we keep only the
    terrain group visible and turn off all other geom groups.

    Set `hide_robot=False` to render the full scene (useful when validating
    real-camera-style inputs).
    """
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    if hide_robot:
        for g in range(len(opt.geomgroup)):
            opt.geomgroup[g] = 1 if g == terrain_group else 0
    return opt


def crop_bottom_center(image, crop_size, bottom_margin=0):
    crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
    src_h, src_w = image.shape[:2]
    top = src_h - int(bottom_margin) - crop_h
    left = (src_w - crop_w) // 2
    if top < 0 or left < 0:
        raise ValueError(
            f"Cannot crop {crop_h}x{crop_w} from source {src_h}x{src_w} "
            f"with bottom_margin={bottom_margin}."
        )
    return image[top:top + crop_h, left:left + crop_w]


def crop_by_pixels(image, crop_pixels):
    crop_left, crop_top, crop_right, crop_bottom = [int(v) for v in crop_pixels]
    src_h, src_w = image.shape[:2]
    top = crop_top
    bottom = src_h - crop_bottom
    left = crop_left
    right = src_w - crop_right
    if top >= bottom or left >= right:
        raise ValueError(
            f"Invalid crop_pixels={crop_pixels} for source {src_h}x{src_w}."
        )
    return image[top:bottom, left:right]


def render_depth(renderer, data, cam_id, render_size, full_size, crop_pixels,
                 policy_size, near_clip, far_clip, scene_option=None, rot90_k=1):
    """Render depth, resize to training full-frame, crop ROI, normalize.

    Sizes are (H, W). Deployment mirrors training by first forming a
    ``full_size`` frame and then applying the configured pixel crop.
    """
    if scene_option is None:
        renderer.update_scene(data, camera=cam_id)
    else:
        renderer.update_scene(data, camera=cam_id, scene_option=scene_option)
    raw = renderer.render()  # positive depth in meters, shape HxW
    if rot90_k:
        raw = np.rot90(raw, k=rot90_k).copy()
    if raw.shape != tuple(render_size):
        raise RuntimeError(
            f"MuJoCo depth shape {raw.shape} does not match configured render_size {tuple(render_size)}."
        )
    if raw.shape != tuple(full_size):
        raw = cv2.resize(
            raw,
            (int(full_size[1]), int(full_size[0])),
            interpolation=cv2.INTER_AREA,
        )
    cropped = crop_by_pixels(raw, crop_pixels)
    if cropped.shape != tuple(policy_size):
        cropped = cv2.resize(
            cropped,
            (int(policy_size[1]), int(policy_size[0])),
            interpolation=cv2.INTER_AREA,
        )
    return normalize_depth(cropped, near_clip, far_clip)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="h1_camera.yaml",
                        help="config name in deploy_camera/deploy_mujoco/configs/")
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", args.config
    )
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    policy_path = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_duration = cfg["simulation_duration"]
    simulation_dt = cfg["simulation_dt"]
    control_decimation = cfg["control_decimation"]

    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)

    ang_vel_scale = float(cfg["ang_vel_scale"])
    dof_pos_scale = float(cfg["dof_pos_scale"])
    dof_vel_scale = float(cfg["dof_vel_scale"])
    action_scale = float(cfg["action_scale"])
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    cmd = np.array(cfg["cmd_init"], dtype=np.float32)

    num_actions = int(cfg["num_actions"])
    num_obs = int(cfg["num_obs"])
    obs_history_len = int(cfg["obs_history_len"])

    # depth config
    depth_far_clip = float(cfg["depth_far_clip"])
    depth_near_clip = float(cfg["depth_near_clip"])
    depth_buffer_len = int(cfg["depth_buffer_len"])
    depth_size = tuple(cfg["depth_size"])  # policy crop size (H, W)
    depth_render_size = tuple(cfg.get("depth_render_size", depth_size))  # full camera frame (H, W)
    depth_full_size = tuple(cfg.get("depth_full_size", depth_render_size))
    depth_crop_pixels = tuple(cfg.get("depth_crop_pixels", [0, 0, 0, 0]))
    depth_vfov = float(cfg.get("depth_vfov", 58.0))
    cam_update_interval = int(cfg["cam_update_interval"])
    cam_name = cfg.get("cam_name", "depth_cam")

    show_depth = bool(cfg.get("show_depth", True))
    # The policy was trained with terrain-only depth (warp renderer skips the
    # robot mesh). Hiding the robot in the depth render keeps deployment
    # in-distribution. Set this to False to simulate a real camera feed.
    hide_robot_in_depth = bool(cfg.get("hide_robot_in_depth", True))
    terrain_group = int(cfg.get("terrain_geom_group", 2))
    depth_rot90_k = int(cfg.get("depth_rot90_k", 1))

    clip_obs = float(cfg.get("clip_observations", 100.0))
    clip_actions = float(cfg.get("clip_actions", 100.0))

    assert num_obs == 39, f"H1 loco policy expects num_obs=39, got {num_obs}"
    assert num_actions == 10, f"H1 loco policy expects num_actions=10, got {num_actions}"
    assert depth_buffer_len >= 1, "buffer_len must be >= 1 to feed the policy"

    # ---- runtime buffers ----
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs_buf = np.zeros(num_obs, dtype=np.float32)
    trajectory_history = torch.zeros(1, obs_history_len, num_obs, dtype=torch.float32)
    dh, dw = depth_size[0], depth_size[1]
    depth_buffer = torch.zeros(1, depth_buffer_len, dh, dw, dtype=torch.float32)
    depth_initialized = False

    # ---- mujoco model ----
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt

    # initial pose: keep floating base at default height with default leg joint angles
    data.qpos[7:7 + num_actions] = default_angles
    mujoco.mj_forward(model, data)

    # ---- camera ----
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise RuntimeError(
            f"Camera '{cam_name}' not found in {xml_path}. "
            f"Make sure the scene XML includes a <camera name=\"{cam_name}\" .../>."
        )
    # If a 90-degree correction is applied, ask MuJoCo for the swapped
    # framebuffer so the corrected image is exactly depth_render_size.  In
    # that case MuJoCo's raw vertical FOV becomes the corrected image's
    # horizontal FOV, so compute the raw fovy from the desired final VFOV.
    if depth_rot90_k % 2:
        renderer_width, renderer_height = depth_render_size[0], depth_render_size[1]
        final_aspect = float(depth_render_size[1]) / float(depth_render_size[0])
        raw_fovy = math.degrees(2.0 * math.atan(final_aspect * math.tan(math.radians(depth_vfov) / 2.0)))
    else:
        renderer_width, renderer_height = depth_render_size[1], depth_render_size[0]
        raw_fovy = depth_vfov
    model.cam_fovy[cam_id] = raw_fovy
    depth_renderer = mujoco.Renderer(model, width=renderer_width, height=renderer_height)
    depth_renderer.enable_depth_rendering()
    depth_scene_option = make_depth_scene_option(hide_robot_in_depth, terrain_group)

    # ---- policy ----
    policy = torch.jit.load(policy_path)
    policy.eval()

    print(f"Loaded policy: {policy_path}")
    print(f"  obs dim: {num_obs}, history: {obs_history_len}x{num_obs}, "
          f"depth: {depth_buffer_len}x{depth_size[0]}x{depth_size[1]} "
          f"from render {depth_render_size[0]}x{depth_render_size[1]} crop, "
          f"vfov={depth_vfov:.1f} deg")
    print(f"Initial cmd (vx, vy, wz) = {cmd.tolist()}")

    counter = 0
    cam_counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # ---- low-level PD: runs every physics step ----
            tau = pd_control(
                target_dof_pos, data.qpos[7:7 + num_actions], kps,
                np.zeros_like(kds), data.qvel[6:6 + num_actions], kds,
            )
            if not np.isfinite(tau).all():
                tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
            data.ctrl[:num_actions] = tau
            mujoco.mj_step(model, data)

            counter += 1
            if counter % control_decimation == 0:
                # ---- proprio observation ----
                qj = data.qpos[7:7 + num_actions]
                dqj = data.qvel[6:6 + num_actions]
                quat = data.qpos[3:7]              # wxyz from mujoco
                omega = data.qvel[3:6]             # body-frame angular velocity

                qj_n = (qj - default_angles) * dof_pos_scale
                dqj_n = dqj * dof_vel_scale
                grav = get_gravity_orientation(quat)
                omega_n = omega * ang_vel_scale

                obs_buf[0:3] = cmd * cmd_scale
                obs_buf[3:6] = omega_n
                obs_buf[6:9] = grav
                obs_buf[9:9 + num_actions] = qj_n
                obs_buf[9 + num_actions:9 + 2 * num_actions] = dqj_n
                obs_buf[9 + 2 * num_actions:9 + 3 * num_actions] = action

                np.clip(obs_buf, -clip_obs, clip_obs, out=obs_buf)
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0)  # (1, 39)

                # ---- update history (chronological, newest at the end) ----
                trajectory_history = torch.cat(
                    [trajectory_history[:, 1:], obs_tensor.unsqueeze(1)], dim=1
                )

                # ---- depth update at lower rate ----
                if cam_counter % cam_update_interval == 0:
                    depth_img = render_depth(
                        depth_renderer, data, cam_id, depth_render_size,
                        depth_full_size, depth_crop_pixels, depth_size,
                        depth_near_clip, depth_far_clip,
                        scene_option=depth_scene_option,
                        rot90_k=depth_rot90_k,
                    )
                    depth_t = torch.from_numpy(depth_img).float()
                    if not depth_initialized:
                        depth_buffer = depth_t.expand(1, depth_buffer_len, dh, dw).clone()
                        depth_initialized = True
                    else:
                        depth_buffer = torch.cat(
                            [depth_buffer[:, 1:], depth_t.unsqueeze(0).unsqueeze(0)], dim=1
                        )
                    if show_depth:
                        cv2.namedWindow("depth_cam", cv2.WINDOW_NORMAL)
                        cv2.imshow("depth_cam", depth_buffer[0, -1].numpy() + 0.5)
                        cv2.waitKey(1)
                cam_counter += 1

                # ---- inference ----
                depth_in = depth_buffer
                with torch.no_grad():
                    action = policy(obs_tensor, trajectory_history, depth_in)
                action = action.detach().cpu().numpy().squeeze(0).astype(np.float32)
                np.clip(action, -clip_actions, clip_actions, out=action)

                target_dof_pos = action * action_scale + default_angles

            viewer.sync()
            sleep_left = model.opt.timestep - (time.time() - step_start)
            if sleep_left > 0:
                time.sleep(sleep_left)


if __name__ == "__main__":
    main()
