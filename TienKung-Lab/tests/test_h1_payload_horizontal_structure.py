from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_h1_payload_horizontal_asset_cfg_declared():
    source = _read("legged_lab/assets/unitree/unitree.py")
    assert "H1_PAYLOAD_HORIZONTAL_CFG" in source
    assert "h1_payload_horizontal.urdf" in source
    assert "payload_mount" in source
    assert "h1_extinguisher_payload" in source


def test_h1_dwaq_uses_payload_asset_and_randomization_ranges():
    source = _read("legged_lab/envs/h1/h1_dwaq_config.py")
    assert "H1_PAYLOAD_HORIZONTAL_CFG" in source
    assert "self.scene.robot = H1_PAYLOAD_HORIZONTAL_CFG" in source
    assert "h1_extinguisher_payload" in source
    assert "mass_distribution_params" in source
    assert "(2.0, 4.0)" in source
    assert "payload_mount_joint_ranges" in source
    assert "(-0.03, 0.03)" in source
    assert "(-0.02, 0.02)" in source
    assert "(-0.12, 0.12)" in source


def test_dwaq_env_keeps_policy_joints_separate_from_payload_mount_joints():
    source = _read("legged_lab/envs/g1/g1_dwaq_env.py")
    assert "controlled_joint_names" in source
    assert "payload_mount_joint_names" in source
    assert "payload_mount_target" in source
    assert "controlled_joint_ids" in source
    assert "payload_mount_joint_ids" in source
    assert "set_joint_position_target(processed_actions, joint_ids=self.controlled_joint_ids)" in source
    assert "set_joint_position_target(self.payload_mount_target, joint_ids=self.payload_mount_joint_ids)" in source


def test_mujoco_horizontal_payload_model_exists():
    path = REPO_ROOT / "legged_lab/assets/h1_description/mjcf/h1_payload_horizontal.xml"
    assert path.exists()
    source = path.read_text(encoding="utf-8")
    assert "h1_extinguisher_payload" in source
    assert "pos=\"-0.18 0 0.10\"" in source
    assert "type=\"cylinder\"" in source
    assert "quat=\"0.70710678 0.70710678 0 0\"" in source


def test_urdf_horizontal_payload_nominal_mount_is_lowered():
    source = _read("legged_lab/assets/h1_description/urdf/h1_payload_horizontal.urdf")
    assert "<origin xyz=\"-0.18 0 0.10\" rpy=\"0 0 0\"/>" in source


def test_sim2sim_lists_payload_scene():
    source = _read("legged_lab/scripts/sim2sim_h1_dwaq.py")
    assert "h1_payload_horizontal.xml" in source
    assert "available_scenes[\"h1_payload\"]" in source
