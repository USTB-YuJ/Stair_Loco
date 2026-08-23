import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stair_nosing_generator_and_config_are_declared():
    source = (REPO_ROOT / "legged_lab/terrains/stair_nosing.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    declared_names = {
        node.name
        for node in module.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }

    assert "pyramid_stairs_with_nosing_terrain" in declared_names
    assert "MeshPyramidStairsWithNosingTerrainCfg" in declared_names
    assert "nosing_depth: float = 0.01" in source
    assert "nosing_thickness: float = 0.01" in source
    assert "range(num_steps + 1)" in source
    assert "nose_thickness = min(cfg.nosing_thickness, step_height)" in source


def test_dwaq_terrain_uses_nosing_on_ascending_stairs():
    source = (REPO_ROOT / "legged_lab/terrains/terrain_generator_cfg.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    dwaq_source = source.split("DWAQ_TERRAINS_CFG =", maxsplit=1)[1].split(
        "DWAQ_HARD_TERRAINS_CFG =", maxsplit=1
    )[0]

    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DWAQ_TERRAINS_CFG" for target in node.targets)
    )
    sub_terrains = next(
        keyword.value
        for keyword in assignment.value.keywords
        if keyword.arg == "sub_terrains"
    )
    proportions = {
        ast.literal_eval(key): ast.literal_eval(
            next(keyword.value for keyword in value.keywords if keyword.arg == "proportion")
        )
        for key, value in zip(sub_terrains.keys, sub_terrains.values)
    }

    assert proportions["stairs_up_nosing_26"] == 0.07
    assert proportions["stairs_up_nosing_30"] == 0.07
    assert proportions["stairs_up_nosing_34"] == 0.06
    assert proportions["stairs_down_26"] == 0.07
    assert proportions["stairs_down_30"] == 0.07
    assert proportions["stairs_down_34"] == 0.06

    assert dwaq_source.count("nosing_depth=0.01") == 3
    assert dwaq_source.count("nosing_thickness=0.01") == 3
    assert abs(
        sum(proportions[name] for name in proportions if name.startswith("stairs_up_")) - 0.20
    ) < 1e-9
    assert abs(
        sum(proportions[name] for name in proportions if name.startswith("stairs_down_")) - 0.20
    ) < 1e-9
    assert abs(sum(proportions.values()) - 1.0) < 1e-9
