from legged_gym import LEGGED_GYM_ROOT_DIR
import cv2
import numpy as np
import os
import threading
import sys
import tty
import termios
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit_depth, task_registry
from legged_gym.utils.depth_roi import crop_window_from_pixels
import torch


# Global flag for stopping recording on keypress
_stop_recording = False

def _keyboard_listener():
    """Thread that waits for 'q' keypress to stop recording."""
    global _stop_recording
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not _stop_recording:
            ch = sys.stdin.read(1)
            if ch == 'q':
                _stop_recording = True
                print('\n[Recording] Stop signal received. Finishing...')
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _safety_bgr(score):
    score = float(np.clip(score, 0.0, 1.0))
    return (0, int(255 * score), int(255 * (1.0 - score)))


def _draw_affordance_matrix(values, title, cell=54):
    values = np.asarray(values, dtype=np.float32)
    rows, cols = values.shape
    top = 26
    img = np.zeros((top + rows * cell, cols * cell, 3), dtype=np.uint8)
    cv2.putText(img, title, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1)
    for r in range(rows):
        for c in range(cols):
            score = float(values[r, c])
            x0, y0 = c * cell, top + r * cell
            color = _safety_bgr(score)
            cv2.rectangle(img, (x0 + 2, y0 + 2), (x0 + cell - 2, y0 + cell - 2), color, -1)
            cv2.rectangle(img, (x0 + 2, y0 + 2), (x0 + cell - 2, y0 + cell - 2), (40, 40, 40), 1)
            txt = f'{score:.2f}'
            cv2.putText(img, txt, (x0 + 8, y0 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return img


def _affordance_head_viz(labels, predictions, candidate_x, candidate_y_offsets):
    if labels is None or predictions is None:
        return None
    labels = labels.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    if labels.ndim != 3 or predictions.ndim != 3:
        return None
    n_y = max(1, len(candidate_y_offsets))
    n_x = max(1, min(len(candidate_x), labels.shape[1] // n_y))
    n = n_x * n_y
    labels = labels[:, :n, 0].reshape(2, n_x, n_y)
    predictions = predictions[:, :n, 0].reshape(2, n_x, n_y)
    blocks = []
    for side, side_name in enumerate(('Left', 'Right')):
        gt_panel = _draw_affordance_matrix(labels[side], f'{side_name} GT')
        pred_panel = _draw_affordance_matrix(predictions[side], f'{side_name} Pred')
        blocks.append(np.hstack((gt_panel, pred_panel)))
    width = max(b.shape[1] for b in blocks)
    blocks = [np.pad(b, ((0, 0), (0, width - b.shape[1]), (0, 0))) for b in blocks]
    return np.vstack(blocks)

def _bev_safety_viz(values, title, scale=8):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu().numpy()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        return None
    values = np.clip(values, 0.0, 1.0)
    green = (values * 255).astype(np.uint8)
    red = ((1.0 - values) * 255).astype(np.uint8)
    blue = np.zeros_like(green, dtype=np.uint8)
    img = np.stack((blue, green, red), axis=-1)
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img

def play(args):
    global _stop_recording
    _stop_recording = False

    # Recording setup
    if args.record:
        record_dir = args.record_path
        if record_dir is None:
            from datetime import datetime
            record_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'recordings',
                                       datetime.now().strftime('%b%d_%H-%M-%S'))
        os.makedirs(record_dir, exist_ok=True)
        print(f'Recording frames to: {record_dir}')
        print('Press [q] to stop recording and save video.')
        # Start keyboard listener thread
        kb_thread = threading.Thread(target=_keyboard_listener, daemon=True)
        kb_thread.start()
    else:
        record_dir = None

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 100
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    env_cfg.terrain.difficulty_level = 1.0
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 8
    env_cfg.domain_rand.push_interval_min_s = 8
    env_cfg.domain_rand.max_push_vel_xy = 1
    env_cfg.domain_rand.min_push_vel_xy = 1

    env_cfg.asset.self_collisions = 0

    env_cfg.env.test = True

    if hasattr(env_cfg, 'footstep_guidance'):
        env_cfg.footstep_guidance.enable_stair_path_guidance = True
        env_cfg.footstep_guidance.enable_footstep_guidance = False
        env_cfg.footstep_guidance.visualize_stair_path = True
        env_cfg.footstep_guidance.visualize_footstep_targets = False
        env_cfg.footstep_guidance.visualize_affordance_candidates = False
        env_cfg.footstep_guidance.visualize_affordance_labels = False
        env_cfg.footstep_guidance.visualize_affordance_predictions = False
        env_cfg.footstep_guidance.save_topdown_debug = False

    env_cfg.depth.y_angle = [48, 48]  # aligned with G1 config (was 70)
    env_cfg.depth.x_angle = [0, 0]
    env_cfg.depth.z_angle = [0, 0]
    env_cfg.depth.x_pos_range = [0, 0]
    env_cfg.depth.y_pos_range = [0, 0]
    env_cfg.depth.z_pos_range = [0, 0]
    env_cfg.depth.use_camera = False
    env_cfg.depth.warp_camera = True
    env_cfg.depth.compute_warp_safety_heatmap = True
    env_cfg.depth.store_warp_depth_debug_stages = True
    env_cfg.depth.dis_noise = 0
    env_cfg.depth.gaussian_noise = False
    env_cfg.depth.gaussian_noise_std = 0.05
    env_cfg.depth.gaussian_filter = False
    env_cfg.depth.gaussian_filter_kernel = [5]
    env_cfg.depth.gaussian_filter_sigma = 1.5

    # Depth viz export (after depth overrides: play uses warp depth path)
    if args.save_depth:
        save_depth_dir = args.save_depth_path
        if save_depth_dir is None:
            from datetime import datetime
            save_depth_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'depth_viz',
                                           datetime.now().strftime('%b%d_%H-%M-%S'))
        os.makedirs(save_depth_dir, exist_ok=True)
        print(f'Saving depth images to: {save_depth_dir}')
    else:
        save_depth_dir = None

    env_cfg.commands.ranges.lin_vel_y = [0, 0]
    env_cfg.commands.ranges.lin_vel_x = [0.7, 0]
    env_cfg.commands.ranges.heading = [0, 0]
    env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 100
    path_guidance_enabled = (hasattr(env_cfg, "footstep_guidance") and (
        getattr(env_cfg.footstep_guidance, "enable_stair_path_guidance", False) or
        getattr(env_cfg.footstep_guidance, "enable_footstep_guidance", False)
    ))
    if path_guidance_enabled:
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.8]
        env_cfg.commands.ranges.lin_vel_y = [-0.5, 0.5]
        env_cfg.commands.ranges.heading = [-3.14, 3.14]
        env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]

    env_cfg.terrain.terrain_dict = {"roughness": 0., 
                                    "slope": 0.,
                                    "pit": 0,
                                    "gap": 0,
                                    "stair": 1}
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.zero_init = False
    train_cfg.runner.load_delta_policy = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # process trajectory history
    num_gait = env.cfg.env.num_gait if hasattr(env.cfg.env, 'num_gait') else 0
    trajectory_history = torch.zeros(size=(env.num_envs, env.obs_history_len, env.num_obs-num_gait), device=env.device)
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs[:, num_gait:].unsqueeze(1)), dim=1)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.load_run, 'exported')
        export_policy_as_jit_depth(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    infos = {}
    if env.cfg.depth.warp_camera:
        infos["depth"] = env.warp_depth_buffer.clone().to(env.device)  
    elif env.cfg.depth.use_camera:
        infos["depth"] = env.depth_buffer.clone().to(env.device)  
    else:
        infos["depth"] = None

    for i in range(int(env.max_episode_length)):
        # get depth image
        if infos["depth"] is not None:
            depth_image = infos['depth']
        if env.cfg.depth.warp_camera or env.cfg.depth.use_camera:
            obs = (obs, depth_image)

        vis_gt_safety = None
        if env.cfg.depth.warp_camera and hasattr(env, 'warp_safety_heatmap_buffer'):
            vis_gt_safety = env.warp_safety_heatmap_buffer[:, -1:].squeeze(1).detach().clone()

        if hasattr(env, '_maybe_update_bev_safety_labels'):
            env._maybe_update_bev_safety_labels(force=True)
        if (getattr(ppo_runner.alg.actor_critic, 'enable_foot_affordance', False)
                and hasattr(env, '_maybe_update_foot_affordance_labels')):
            env._maybe_update_foot_affordance_labels(force=True)

        if isinstance(obs, tuple):
            actions = policy(obs[0].detach(), trajectory_history.detach(), obs[1].detach())
        else:
            actions = policy(obs.detach(), trajectory_history)

        bev_pred = getattr(ppo_runner.alg.actor_critic, '_last_bev_safety_pred', None)
        if bev_pred is not None and getattr(env, 'bev_safety_labels', None) is not None:
            gt_panel = _bev_safety_viz(env.bev_safety_labels[0, 0], 'BEV Safety GT')
            pred_panel = _bev_safety_viz(bev_pred[0], 'BEV Safety Pred')
            if gt_panel is not None and pred_panel is not None:
                cv2.imshow('BEV Safety GT', gt_panel)
                cv2.imshow('BEV Safety Pred', pred_panel)
                cv2.waitKey(1)

        affordance_pred = getattr(ppo_runner.alg.actor_critic, '_last_affordance_pred', None)
        if hasattr(env, 'set_foot_affordance_predictions'):
            env.set_foot_affordance_predictions(affordance_pred)
        if affordance_pred is not None and hasattr(env, 'foot_affordance_labels'):
            fg_cfg = getattr(env.cfg, 'footstep_guidance', None)
            candidate_x = getattr(fg_cfg, 'affordance_candidate_x', [0.10, 0.25, 0.40, 0.55, 0.70]) if fg_cfg is not None else [0.10, 0.25, 0.40, 0.55, 0.70]
            candidate_y_offsets = getattr(fg_cfg, 'affordance_candidate_y_offsets', [-0.08, 0.0, 0.08]) if fg_cfg is not None else [-0.08, 0.0, 0.08]
            panel = _affordance_head_viz(env.foot_affordance_labels[0], affordance_pred[0], candidate_x, candidate_y_offsets)
            if panel is not None:
                cv2.imshow('Foot Affordance Head', panel)
                cv2.waitKey(1)

        vis_pred_safety = None
        pred_masks = getattr(ppo_runner.alg.actor_critic, "_last_pred_masks", None)
        if pred_masks is not None:
            vis_pred_safety = pred_masks[0:1, -1].detach().clone()

        obs, _, _, dones, infos, *_= env.step(actions.detach())

        # Capture frame for recording
        if record_dir is not None and not _stop_recording:
            frame_path = os.path.join(record_dir, f'frame_{i:06d}.png')
            env.gym.write_viewer_image_to_file(env.viewer, frame_path)

        # Noise stage visualization window
        if env.cfg.depth.warp_camera and hasattr(env, '_depth_stage_raw'):
            def _to_vis(t, scale=4, mode="inferno"):
                arr = t[0].cpu().numpy().astype(np.float32)
                if mode == "safety":
                    arr = np.clip(arr, 0.0, 1.0)
                    g = (arr * 255).astype(np.uint8)
                    r = ((1.0 - arr) * 255).astype(np.uint8)
                    b = np.zeros_like(g, dtype=np.uint8)
                    col = np.stack([b, g, r], axis=-1)
                else:
                    mn, mx = arr.min(), arr.max()
                    if mx - mn > 1e-6:
                        arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
                    else:
                        arr = np.zeros_like(arr, dtype=np.uint8)
                    col = cv2.applyColorMap(arr, cv2.COLORMAP_INFERNO)
                return cv2.resize(col, (col.shape[1]*scale, col.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
            raw_safety = getattr(env, '_warp_safety_raw_full', None)
            valid_mask = getattr(env, '_warp_safety_valid_full', None)
            final_safety = getattr(env, '_warp_safety_final_full', None)
            stages = [
                # ("1.raw",        env._depth_stage_raw),
                # ("2.+gaussian",  env._depth_stage_gaussian),
                # ("3.+dis",       env._depth_stage_dis),
                # ("4.+edge",      env._depth_stage_edge),
                # ("5.+patch",     env._depth_stage_patch),
                # ("6.+discont",   env._depth_stage_discontinuity),
                # ("7.normalized", env._depth_stage_normalized),
                # ("8.+filter",    env._raw_warp_depth),
                # ("9.final",      env.warp_depth_buffer[:, -1:].squeeze(1)),
                ("Raw Safety", raw_safety if raw_safety is not None else (vis_gt_safety if vis_gt_safety is not None else env.warp_safety_heatmap_buffer[:, -1:].squeeze(1))),
                ("Valid Mask", valid_mask if valid_mask is not None else torch.ones_like(vis_gt_safety if vis_gt_safety is not None else env.warp_safety_heatmap_buffer[:, -1:].squeeze(1))),
                ("Final Safety", final_safety if final_safety is not None else (vis_gt_safety if vis_gt_safety is not None else env.warp_safety_heatmap_buffer[:, -1:].squeeze(1))),
                ("Pred Safety", vis_pred_safety if vis_pred_safety is not None else torch.zeros(1, *env.cfg.depth.resized, device=env.device)),
            ]
            # debug: first 5 steps only
            import __main__ as _m
            if not hasattr(_m, '_heatmap_checked'):
                _m._heatmap_checked = True
                _buf = vis_gt_safety if vis_gt_safety is not None else env.warp_safety_heatmap_buffer[:, -1]
                _d = env._depth_stage_normalized
                print(f"[DEBUG] safety: min={_buf.min().item():.4f} max={_buf.max().item():.4f} mean={_buf.mean().item():.4f} nonzero={(_buf>1e-6).float().mean().item():.3f}")
                print(f"[DEBUG] depth_norm: min={_d.min().item():.4f} max={_d.max().item():.4f} mean={_d.mean().item():.4f}")
                _w2c = env._warp2cam if hasattr(env, '_warp2cam') else 'no_attr'
                print(f"[DEBUG] _warp2cam: {_w2c is not None if hasattr(env,'_warp2cam') else 'NA'}, height_samples: {env.height_samples is not None}")
            panels = []
            for label, t in stages:
                mode = "safety" if ("Safety" in label or "Valid" in label) else "inferno"
                p = _to_vis(t, mode=mode)
                cv2.putText(p, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                panels.append(p)
            # pad panels to same height before hstack
            max_h = max(p.shape[0] for p in panels)
            padded = [np.pad(p, ((0, max_h-p.shape[0]),(0,0),(0,0))) for p in panels]
            cv2.imshow("Depth Noise Stages", np.hstack(padded))
            cv2.waitKey(1)

        # Save depth image: full-res warp output + red box = region fed to policy when crop_depth True
        if save_depth_dir is not None and hasattr(env, '_raw_warp_depth') and env._raw_warp_depth is not None:
            raw_depth = env._raw_warp_depth[0].cpu().numpy()  # [H, W] for env 0 (same H,W as cfg.depth.original)
            h, w = raw_depth.shape

            # Normalize to 0-255 for visualization
            d_min, d_max = raw_depth.min(), raw_depth.max()
            if d_max - d_min > 1e-6:
                depth_viz = ((raw_depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_viz = np.zeros_like(raw_depth, dtype=np.uint8)
            depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)

            # Match legged_robot.warp_update_depth_buffer crop slice (only when crop is actually applied)
            if env.cfg.depth.crop_depth:
                crop_top, crop_bottom, crop_left, crop_right = crop_window_from_pixels(
                    raw_depth.shape, env.cfg.depth.crop_pixels
                )
                # Tensor rows [top : bottom], cols [left : right] (end exclusive)
                x1, y1 = int(crop_left), int(crop_top)
                x2, y2 = int(crop_right - 1), int(crop_bottom - 1)
                cv2.rectangle(depth_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
                left_label = 'Full depth 96x128 (red = policy ROI)'
            else:
                left_label = 'Warp depth (full frame = policy input; crop_depth off)'

            # Processed tensor after optional crop+resize (same as buffer / policy)
            proc_depth = env.warp_depth_buffer[0, -1].cpu().numpy()
            d_min2, d_max2 = proc_depth.min(), proc_depth.max()
            if d_max2 - d_min2 > 1e-6:
                proc_viz = ((proc_depth - d_min2) / (d_max2 - d_min2) * 255).astype(np.uint8)
            else:
                proc_viz = np.zeros_like(proc_depth, dtype=np.uint8)
            proc_color = cv2.applyColorMap(proc_viz, cv2.COLORMAP_INFERNO)

            ph, pw = proc_depth.shape
            proc_color_resized = cv2.resize(proc_color, (w, h), interpolation=cv2.INTER_NEAREST)
            right_label = f'Policy input {ph}x{pw} (nearest upsample)'

            cv2.putText(depth_color, left_label, (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(proc_color_resized, right_label, (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            combined = np.hstack((depth_color, proc_color_resized))
            viz_scale = 4
            combined = cv2.resize(
                combined,
                (combined.shape[1] * viz_scale, combined.shape[0] * viz_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            depth_path = os.path.join(save_depth_dir, f'depth_{i:06d}.png')
            cv2.imwrite(depth_path, combined)

        # process trajectory history
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        trajectory_history = torch.concat((trajectory_history[:, 1:], obs[:, num_gait:].unsqueeze(1)), dim=1)

        if _stop_recording:
            break

    # Compile frames into video and clean up PNGs
    if record_dir is not None:
        print('Compiling frames into video...')
        import glob
        frame_files = sorted(glob.glob(os.path.join(record_dir, 'frame_*.png')))
        if frame_files:
            first = cv2.imread(frame_files[0])
            h, w = first.shape[:2]
            video_path = os.path.join(record_dir, 'playback.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
            for fname in frame_files:
                img = cv2.imread(fname)
                writer.write(img)
            writer.release()
            print(f'Video saved to: {video_path}')
            # Delete PNG frames

            for fname in frame_files:
                os.remove(fname)
            print(f'Cleaned up {len(frame_files)} PNG frames.')
        else:
            print('Warning: No frames captured')

if __name__ == '__main__':
    EXPORT_POLICY = False
    args = get_args()
    # args.task = "h1_loco"
    args.task = "g1_16dof_loco"
    args.load_run = "/root/gpufree-data/workspace/more/logs/g1_16dof_loco/May19_23-55-37_"
    args.record = True
    args.headless = not args.record  # need viewer for recording
    args.save_depth = False
    play(args)
