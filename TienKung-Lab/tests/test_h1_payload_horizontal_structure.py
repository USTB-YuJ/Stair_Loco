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
    assert "pos=\"-0.18 0 0.05\"" in source
    assert "type=\"box\"" in source
    assert "quat=\"1 0 0 0\"" in source


def test_urdf_horizontal_payload_nominal_mount_is_lowered_to_5cm():
    source = _read("legged_lab/assets/h1_description/urdf/h1_payload_horizontal.urdf")
    assert "<origin xyz=\"-0.18 0 0.05\" rpy=\"0 0 0\"/>" in source


def test_sim2sim_lists_payload_scene():
    source = _read("legged_lab/scripts/sim2sim_h1_dwaq.py")
    assert "h1_payload_horizontal.xml" in source
    assert "available_scenes[\"h1_payload\"]" in source
    assert "available_scenes[\"h1_payload_stairs\"]" in source

def test_sim2sim_payload_stair_scene_and_default_script():
    scene_path = REPO_ROOT / "legged_lab/assets/h1_description/mjcf/scene_payload_horizontal.xml"
    assert scene_path.exists()
    scene = scene_path.read_text(encoding="utf-8")
    assert '<include file="h1_payload_horizontal.xml" />' in scene
    assert '<include file="h1.xml" />' not in scene
    assert 'name="floor"' in scene
    assert 'type="box"' in scene

    sim2sim = _read("sim2sim.sh")
    assert "scene_payload_horizontal.xml" in sim2sim

def test_horizontal_payload_dimensions_match_405_by_62_by_62_mm_box():
    urdf = _read("legged_lab/assets/h1_description/urdf/h1_payload_horizontal.urdf")
    assert '<box size="0.062 0.405 0.062"/>' in urdf
    assert '<inertia ixx="0.04196725" ixy="0" ixz="0" iyy="0.001922" iyz="0" izz="0.04196725"/>' in urdf

    mjcf = _read("legged_lab/assets/h1_description/mjcf/h1_payload_horizontal.xml")
    assert 'size="0.031 0.2025 0.031"' in mjcf
    assert 'diaginertia="0.04196725 0.001922 0.04196725"' in mjcf
