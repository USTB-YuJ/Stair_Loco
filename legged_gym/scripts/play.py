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

    env_cfg.depth.y_angle = [45, 45]
    env_cfg.depth.x_angle = [0, 0]
    env_cfg.depth.z_angle = [0, 0]
    env_cfg.depth.x_pos_range = [0, 0]
    env_cfg.depth.y_pos_range = [0, 0]
    env_cfg.depth.z_pos_range = [0, 0]
    env_cfg.depth.use_camera = False
    env_cfg.depth.warp_camera = True
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

    env_cfg.terrain.terrain_dict = {"roughness": 0., 
                                    "slope": 0.,
                                    "pit": 0,
                                    "gap": 0,
                                    "stair": 1,}
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

        if isinstance(obs, tuple):
            actions = policy(obs[0].detach(), trajectory_history.detach(), obs[1].detach())
        else:
            actions = policy(obs.detach(), trajectory_history)
        obs, _, _, dones, infos, *_= env.step(actions.detach())

        # Capture frame for recording
        if record_dir is not None and not _stop_recording:
            frame_path = os.path.join(record_dir, f'frame_{i:06d}.png')
            env.gym.write_viewer_image_to_file(env.viewer, frame_path)

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
    EXPORT_POLICY = True
    args = get_args()
    args.task = "h1_loco"
    args.load_run = "/root/gpufree-data/workspace/more/logs/h1_loco/May10_00-02-19_"
    args.record = True
    args.headless = not args.record  # need viewer for recording
    args.save_depth = True
    play(args)
