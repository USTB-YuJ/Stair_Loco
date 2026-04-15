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
