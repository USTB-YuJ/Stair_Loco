import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _literal_assignment(module: ast.Module, name: str):
    assignment = next(
        node
        for node in module.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    )
    return ast.literal_eval(assignment.value)


def test_three_step_geometry_matrix_and_exact_distribution_are_declared():
    source = _source("legged_lab/terrains/three_step_corridor.py")
    module = ast.parse(source)
    heights = _literal_assignment(module, "THREE_STEP_HEIGHTS")
    depths = _literal_assignment(module, "THREE_STEP_TREAD_DEPTHS")

    assert heights == (0.14, 0.15, 0.16, 0.17)
    assert depths == (0.28, 0.30, 0.32, 0.34, 0.36)
    assert len(heights) * len(depths) == 20
    assert "proportion=0.035" in source
    assert source.count("MeshFlatCorridorTerrainCfg(proportion=0.10)") == 3
    assert "num_rows=1" in source
    assert "num_cols=200" in source
    assert "size=(5.0, 2.0)" in source


def test_every_step_has_an_exact_one_centimeter_nose():
    source = _source("legged_lab/terrains/three_step_corridor.py")
    assert "for level in range(1, 4):" in source
    assert "nosing_depth: float = 0.01" in source
    assert "nosing_thickness: float = 0.01" in source
    assert 'kind="stair_nosing"' in source
    assert "(cfg.nosing_depth, cfg.size[1], nose_thickness)" in source


def test_three_step_commands_and_success_thresholds_are_task_bound():
    source = _source("legged_lab/envs/h1/h1_three_step_env.py")
    cfg_source = _source("legged_lab/envs/h1/h1_three_step_config.py")

    assert "forward_velocity_range: tuple[float, float] = (0.7, 1.0)" in cfg_source
    assert "turn_yaw_abs_range: tuple[float, float] = (0.4, 0.8)" in cfg_source
    assert "self.vel_command_b[env_ids] = 0.0" in source
    assert "self.vel_command_b[forward_ids, 0].uniform_" in source
    assert "self.vel_command_b[turn_ids, 2] = magnitudes * signs" in source
    assert "approach_distance + 3.0 * tread_depth + 0.10" in source
    assert "flat_forward_success_distance: float = 2.18" in cfg_source
    assert "time_out_buf = base_timeout & ~success" in source


def test_three_step_task_is_isolated_and_registered():
    env_init = _source("legged_lab/envs/__init__.py")
    standard_train = _source("train.sh")
    task_train = _source("train_three_step.sh")

    assert '"h1_dwaq_three_step"' in env_init
    assert "H1ThreeStepDwaqEnv" in env_init
    assert "--task=h1_dwaq_three_step" in task_train
    assert "--task=h1_dwaq" in standard_train
    assert "h1_dwaq_three_step" not in standard_train

