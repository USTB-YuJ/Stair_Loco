from pathlib import Path
from typing import Union

import numpy as np
import yaml

# 本文件所在目录即项目根目录（与 configs/、weights/ 同级）
PROJECT_ROOT = Path(__file__).resolve().parent


class CameraConfig:
    def __init__(self, file_path: Union[str, Path]) -> None:
        self.config_path = Path(file_path).resolve()
        with open(self.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = config.get("weak_motor", [])

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            raw_policy = config["policy_path"]
            if isinstance(raw_policy, str):
                raw_policy = raw_policy.replace("{PROJECT_ROOT}", str(PROJECT_ROOT))
            p = Path(raw_policy)
            self.policy_path = str(p if p.is_absolute() else PROJECT_ROOT / p)

            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            self.arm_waist_kps = config["arm_waist_kps"]
            self.arm_waist_kds = config["arm_waist_kds"]
            self.arm_waist_target = np.array(
                config["arm_waist_target"], dtype=np.float32
            )

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            self.obs_history_len = config["obs_history_len"]
            self.depth_buffer_len = config["depth_buffer_len"]
            self.cam_update_interval = config["cam_update_interval"]
            self.depth_near_clip = config["depth_near_clip"]
            self.depth_far_clip = config["depth_far_clip"]
            self.depth_height = config["depth_height"]
            self.depth_width = config["depth_width"]
            self.depth_rot90_k = config["depth_rot90_k"]
            # HW: policy depth 为 [..., H=48, W=64]（与文档一致）；WH: [..., 64, 48]
            self.depth_tensor_layout = config.get("depth_tensor_layout", "HW")
