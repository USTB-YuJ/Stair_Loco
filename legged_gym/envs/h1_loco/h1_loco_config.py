from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1_Loco_Cfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]

        random_default_pos = False
        # default_joint_angles = {  # = target angles [rad] when action = 0.0
        #     "left_hip_yaw_joint": 0.0,
        #     "left_hip_roll_joint": 0.0,
        #     "left_hip_pitch_joint": -0.15,
        #     "left_knee_joint": 0.35,
        #     "left_ankle_joint": -0.2,
        #     "right_hip_yaw_joint": 0.0,
        #     "right_hip_roll_joint": 0.0,
        #     "right_hip_pitch_joint": -0.15,
        #     "right_knee_joint": 0.35,
        #     "right_ankle_joint": -0.2,
        # }

        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_joint' : -0.2,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_joint' : -0.2,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        hf2mesh_method = "grid"
        max_error = 0.1

        edge_width_thresh = 0
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 5
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True
        min_difficulty_level = 0.5
        test_mode = False

        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = True
        measured_points_x = [
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        measure_horizontal_noise = 0.0

        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 14
        terrain_width = 4
        num_rows = 10
        num_cols = 30

        terrain_dict = {
            "roughness": 1.0,
            "slope": 1.0,
            "pit": 1,
            "gap": 1,
            "stair": 1,
        }
        terrain_proportions = list(terrain_dict.values())

        slope_treshold = 0.75
        origin_zero_z = False
        num_goals = 8
        difficulty_level = 1
        test_mode = False

    class env(LeggedRobotCfg.env):
        num_actions = 10
        num_observations = 9 + 3 * num_actions

        # AMP-related fields are kept for compatibility with AMPOnPolicyRunnerMulti,
        # but the default PPO config sets use_amp=False.
        amp_motion_files = ""
        num_amp_obs = num_actions

        reference_state_initialization = False
        reference_state_initialization_prob = 0.0

        feet_info = True
        priv_info = True
        foot_force_info = True
        scan_dot = True

        num_privileged_obs = 12 + 3 * num_actions
        if feet_info:
            num_privileged_obs += 12
        if priv_info:
            # base height (1) + friction (1) + added mass (1) + com pos (3) + PD gains delta (2*num_actions)
            num_privileged_obs += 6 + 2 * num_actions
        if foot_force_info:
            num_privileged_obs += 6
        if scan_dot:
            num_privileged_obs += 187

    class depth(LeggedRobotCfg.depth):
        use_camera = False
        warp_camera = True
        warp_device = "cuda:0"

        position = [0.0576235, 0.01753, 0.42987]
        original = (64, 64)
        resized = (64, 64)
        near_clip = 0
        far_clip = 2
        update_interval = 5
        buffer_len = 2 + 1
        fovy_range = [79.3, 79.3]

        y_angle = [42, 48]
        z_angle = [-1, 1]
        x_angle = [-1, 1]

        rand_position = False
        x_pos_range = [-0.01, 0.01]
        y_pos_range = [-0.01, 0.01]
        z_pos_range = [-0.01, 0.01]

        dis_noise = 0

        gaussian_noise = False
        gaussian_noise_std = 0.05

        gaussian_filter = False
        gaussian_filter_kernel = [1, 3, 5]
        gaussian_filter_sigma = 1.2

        random_cam_delay = False

        add_body_mask = False
        body_mask_path = "./body_mask_data/body_masks_real+sim.npz"
        crop_depth = False
        crop_pixels = [10, 20, 10, 5]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 2]
        randomize_base_mass = True
        added_mass_range = [-5.0, 5.0]
        push_robots = True
        push_interval_s = 8
        push_interval_min_s = 8
        min_push_vel_xy = 1
        max_push_vel_xy = 1
        stair_no_push = False

        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]

        randomize_com_pos = True
        com_x_pos_range = [-0.03, 0.03]
        com_y_pos_range = [-0.03, 0.03]
        com_z_pos_range = [-0.03, 0.03]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_restitution = True
        restitution_range = [0.0, 1.0]

        randomize_actuation_offset = True
        actuation_offset_range = [-0.1, 0.1]

        delay_update_global_steps = 24 * 5000
        action_delay = True
        action_curr_step = [0, 1, 2]
        action_buf_len = 5

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class control(LeggedRobotCfg.control):
        control_type = "P"
        # stiffness = {
        #     "hip_yaw_joint": 100.0,
        #     "hip_roll_joint": 100.0,
        #     "hip_pitch_joint": 100.0,
        #     "knee_joint": 150.0,
        #     "ankle_joint": 40.0,
        # }
        # damping = {
        #     "hip_yaw_joint": 2.0,
        #     "hip_roll_joint": 2.0,
        #     "hip_pitch_joint": 2.0,
        #     "knee_joint": 4.0,
        #     "ankle_joint": 2.0,
        # }

        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 200,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 150,
                     "elbow":100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf"
        name = "h1"

        # H1 has fixed joints (e.g. torso) that may be collapsed by default, which can hide expected body names.
        collapse_fixed_joints = False
        torso_name = "torso_link"

        foot_name = "ankle_link"
        knee_name = "knee_link"

        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False

        # used by optional foot-edge reward/visualization
        feet_indicator_offset = [
            [-0.04, 0.0, -0.04],
            [-0.02, 0.0, -0.04],
            [0.0, 0.0, -0.04],
            [0.02, 0.0, -0.04],
            [0.06, 0.0, -0.04],
            [0.1, 0.0, -0.04],
        ]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        feet_min_lateral_distance_target = 0.14
        clearance_height_target = -0.6

        class scales:
            tracking_lin_vel = 2.0
            tracking_ang_vel = 2.0

            dof_acc = -5e-7
            dof_vel = -1e-3
            action_rate = -0.03
            action_smoothness = -0.05

            ang_vel_xy = -0.05
            orientation = -2.0
            joint_power = -2.5e-5
            feet_clearance = -0.25
            feet_stumble = -1.0
            torques = -1e-5

            # G1-only terms (arms/extra hips) disabled for H1
            arm_joint_deviation = 0.0
            hip_joint_deviation = 0.0

            dof_pos_limits = -2.0
            dof_vel_limits = -1.0
            torque_limits = -1.0
            no_fly = 0.25
            feet_lateral_distance = 0.5
            feet_slippage = -0.25
            feet_contact_force = -2.5e-4
            feet_force_rate = -2.5e-4
            feet_contact_momentum = -2.5e-4
            collision = -15.0
            feet_air_time = 1.0
            stuck = -1.0
            cheat = -2.0
            feet_edge = -0.5
            y_offset_pen = -0.5

        feet_contact_force_range = [200.0, 600.0]


class H1_Loco_CfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "AMPOnPolicyRunnerMulti"

    class policy:
        init_noise_std = 0.8
        activation = "elu"
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        his_latent_dim = 64

    class algorithm(LeggedRobotCfgPPO.algorithm):
        use_amp = False

        # kept for compatibility if you later turn AMP on
        amp_loader_type = "lafan_16dof_multi"
        amp_loader_class_name = "G1_AMPLoader"

        entropy_coef = 0.01
        policy_learning_rate = 5e-4

    class runner(LeggedRobotCfgPPO.runner):
        num_amp_frames = 5
        policy_class_name = "ActorCriticDepth"
        algorithm_class_name = "AMPPPOMulti"

        use_lerp = False
        max_iterations = 50000
        run_name = ""
        experiment_name = "h1_loco"
        save_interval = 500

        amp_reward_coef = 5
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.5
        amp_discr_hidden_dims = [1024, 512]
