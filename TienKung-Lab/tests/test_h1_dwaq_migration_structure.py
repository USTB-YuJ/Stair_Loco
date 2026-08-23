import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _assigned_names(module: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
    return names


def test_h1_dwaq_asset_and_config_files_exist():
    expected_paths = [
        REPO_ROOT / "legged_lab/assets/h1_description/mjcf/h1.xml",
        REPO_ROOT / "legged_lab/assets/h1_description/urdf/h1.urdf",
        REPO_ROOT / "legged_lab/envs/h1/h1_config.py",
        REPO_ROOT / "legged_lab/envs/h1/h1_dwaq_config.py",
        REPO_ROOT / "legged_lab/scripts/sim2sim_h1_dwaq.py",
    ]

    for path in expected_paths:
        assert path.exists(), f"Missing expected migration file: {path}"


def test_h1_cfg_is_declared_in_unitree_assets():
    module = _parse_module(REPO_ROOT / "legged_lab/assets/unitree/unitree.py")
    assert "H1_CFG" in _assigned_names(module)


def test_h1_usd_path_is_declared():
    unitree_path = REPO_ROOT / "legged_lab/assets/unitree/unitree.py"
    source = unitree_path.read_text(encoding="utf-8")
    assert "unitree/h1/h1.usd" in source


def test_h1_dwaq_config_classes_exist():
    module = _parse_module(REPO_ROOT / "legged_lab/envs/h1/h1_dwaq_config.py")
    names = _assigned_names(module)
    assert {"H1DwaqRewardCfg", "H1DwaqEnvCfg", "H1DwaqAgentCfg"} <= names


def test_h1_dwaq_task_is_registered():
    module = _parse_module(REPO_ROOT / "legged_lab/envs/__init__.py")
    registered_tasks: set[str] = set()

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "task_registry" or node.func.attr != "register":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            continue
        registered_tasks.add(node.args[0].value)

    assert "h1_dwaq" in registered_tasks


def test_h1_dwaq_payload_reward_weights_are_declared():
    source = (REPO_ROOT / "legged_lab/envs/h1/h1_dwaq_config.py").read_text(encoding="utf-8")

    assert "upright_posture_reward = RewTerm(" in source
    assert "weight=2.5" in source
    assert (
        "ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)"
        in source
    )

def test_h1_dwaq_feet_orientation_reward_is_declared():
    rewards_source = (REPO_ROOT / "legged_lab/mdp/rewards.py").read_text(encoding="utf-8")
    h1_dwaq_source = (REPO_ROOT / "legged_lab/envs/h1/h1_dwaq_config.py").read_text(encoding="utf-8")

    assert "def feet_orientation_l2(" in rewards_source
    assert "feet_orientation_l2 = RewTerm(" in h1_dwaq_source
    assert "func=mdp.feet_orientation_l2" in h1_dwaq_source
    assert 'body_names=".*ankle.*"' in h1_dwaq_source
    assert "weight=-0.25" in h1_dwaq_source


def test_h1_dwaq_uses_lower_terrain_relative_base_height_reward():
    source = (REPO_ROOT / "legged_lab/envs/h1/h1_dwaq_config.py").read_text(encoding="utf-8")
    rewards_source = (REPO_ROOT / "legged_lab/mdp/rewards.py").read_text(encoding="utf-8")

    assert "base_height = RewTerm(" in source
    assert "func=mdp.base_height" in source
    assert '"target_height": 0.95' in source
    assert '"sensor_cfg": SceneEntityCfg("height_scanner")' in source
    assert '"z": (-0.05, 0.05)' in source
    assert "finite_hits = torch.isfinite(terrain_z)" in rewards_source
    assert "valid_hit_count.clamp(min=1)" in rewards_source
    assert "height_error = torch.nan_to_num(" in rewards_source


def test_h1_dwaq_freezes_gait_phase_for_standing_commands():
    base_source = (REPO_ROOT / "legged_lab/envs/base/base_config.py").read_text(encoding="utf-8")
    env_source = (REPO_ROOT / "legged_lab/envs/g1/g1_dwaq_env.py").read_text(encoding="utf-8")
    config_source = (REPO_ROOT / "legged_lab/envs/h1/h1_dwaq_config.py").read_text(encoding="utf-8")
    sim2sim_source = (REPO_ROOT / "legged_lab/scripts/sim2sim_h1_dwaq.py").read_text(encoding="utf-8")

    assert "standing_command_threshold: float = 0.1" in base_source
    assert "moving_mask = command_magnitude >= standing_threshold" in env_source
    assert "self.phase = torch.where(moving_mask, walking_phase" in env_source
    assert "self.robot.gait_phase.standing_command_threshold = 0.1" in config_source
    assert "command_magnitude < self.cfg.gait_phase.standing_command_threshold" in sim2sim_source
    assert "self.gait_phase_time = 0.0" in sim2sim_source
