"""Task-specific three-step corridor terrains for H1 blind stair climbing."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


THREE_STEP_HEIGHTS = (0.14, 0.15, 0.16, 0.17)
THREE_STEP_TREAD_DEPTHS = (0.28, 0.30, 0.32, 0.34, 0.36)
THREE_STEP_APPROACH_DISTANCE = 1.0
THREE_STEP_FLAT_FORWARD_DISTANCE = 2.18


def _box(dimensions: tuple[float, float, float], center: tuple[float, float, float], **metadata):
    mesh = trimesh.creation.box(
        dimensions,
        transform=trimesh.transformations.translation_matrix(center),
    )
    mesh.metadata.update(metadata)
    return mesh


def three_step_corridor_terrain(
    difficulty: float,
    cfg: MeshThreeStepCorridorTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build a straight three-step staircase with an exact projecting nose."""
    del difficulty
    if cfg.step_height <= 0.0 or cfg.tread_depth <= 0.0:
        raise ValueError("step_height and tread_depth must be positive")
    if cfg.nosing_depth <= 0.0 or cfg.nosing_thickness <= 0.0:
        raise ValueError("nosing_depth and nosing_thickness must be positive")

    first_riser_x = cfg.spawn_x + cfg.approach_distance
    third_tread_end_x = first_riser_x + 3.0 * cfg.tread_depth
    if third_tread_end_x >= cfg.size[0]:
        raise ValueError(
            f"Three-step staircase ends at x={third_tread_end_x:.3f}, outside tile length {cfg.size[0]:.3f}"
        )

    meshes = [
        _box(
            (cfg.size[0], cfg.size[1], cfg.base_thickness),
            (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * cfg.base_thickness),
            kind="corridor_base",
        )
    ]

    for level in range(1, 4):
        tread_start = first_riser_x + (level - 1) * cfg.tread_depth
        tread_end = first_riser_x + level * cfg.tread_depth
        if level == 3:
            tread_end = cfg.size[0]

        top_height = level * cfg.step_height
        tread_length = tread_end - tread_start
        meshes.append(
            _box(
                (tread_length, cfg.size[1], top_height),
                (0.5 * (tread_start + tread_end), 0.5 * cfg.size[1], 0.5 * top_height),
                kind="stair_step",
                level=level,
                top_height=top_height,
            )
        )

        nose_thickness = min(cfg.nosing_thickness, cfg.step_height)
        meshes.append(
            _box(
                (cfg.nosing_depth, cfg.size[1], nose_thickness),
                (
                    tread_start - 0.5 * cfg.nosing_depth,
                    0.5 * cfg.size[1],
                    top_height - 0.5 * nose_thickness,
                ),
                kind="stair_nosing",
                level=level,
                projection=cfg.nosing_depth,
            )
        )

    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], 0.0], dtype=np.float64)
    return meshes, origin


def flat_corridor_terrain(
    difficulty: float,
    cfg: MeshFlatCorridorTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build a flat tile with the same footprint and spawn convention."""
    del difficulty
    mesh = _box(
        (cfg.size[0], cfg.size[1], cfg.base_thickness),
        (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * cfg.base_thickness),
        kind="corridor_base",
    )
    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], 0.0], dtype=np.float64)
    return [mesh], origin


@configclass
class MeshThreeStepCorridorTerrainCfg(SubTerrainBaseCfg):
    """Configuration for one fixed-size three-step corridor."""

    function = three_step_corridor_terrain
    step_height: float = 0.15
    tread_depth: float = 0.30
    approach_distance: float = THREE_STEP_APPROACH_DISTANCE
    spawn_x: float = 0.75
    nosing_depth: float = 0.01
    nosing_thickness: float = 0.01
    base_thickness: float = 0.10


@configclass
class MeshFlatCorridorTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a flat task tile sharing the stair spawn layout."""

    function = flat_corridor_terrain
    spawn_x: float = 0.75
    base_thickness: float = 0.10


def _make_three_step_sub_terrains() -> dict[str, SubTerrainBaseCfg]:
    sub_terrains: dict[str, SubTerrainBaseCfg] = {}
    for height in THREE_STEP_HEIGHTS:
        for tread_depth in THREE_STEP_TREAD_DEPTHS:
            height_cm = round(height * 100)
            depth_cm = round(tread_depth * 100)
            sub_terrains[f"stairs_h{height_cm}_d{depth_cm}"] = MeshThreeStepCorridorTerrainCfg(
                proportion=0.035,
                step_height=height,
                tread_depth=tread_depth,
            )

    sub_terrains["flat_forward"] = MeshFlatCorridorTerrainCfg(proportion=0.10)
    sub_terrains["flat_turn"] = MeshFlatCorridorTerrainCfg(proportion=0.10)
    sub_terrains["flat_stand"] = MeshFlatCorridorTerrainCfg(proportion=0.10)
    return sub_terrains


THREE_STEP_CORRIDOR_TERRAINS_CFG = TerrainGeneratorCfg(
    # One row prevents terrain-level progression. Curriculum generation is
    # used only to allocate the 200 columns deterministically by proportion.
    curriculum=True,
    size=(5.0, 2.0),
    border_width=1.0,
    border_height=0.10,
    num_rows=1,
    num_cols=200,
    horizontal_scale=0.01,
    vertical_scale=0.001,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains=_make_three_step_sub_terrains(),
)

