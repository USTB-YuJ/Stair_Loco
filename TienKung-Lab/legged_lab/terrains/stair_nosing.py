# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom stair terrains used by TienKung-Lab."""

from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.trimesh.mesh_terrains import pyramid_stairs_terrain
from isaaclab.utils import configclass


def pyramid_stairs_with_nosing_terrain(
    difficulty: float,
    cfg: MeshPyramidStairsWithNosingTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate pyramid stairs with a thin outward-projecting nose on every riser.

    Each nose occupies the top ``nosing_thickness`` of its riser and projects
    ``nosing_depth`` toward the adjacent lower tread. The step top and the
    terrain's total height are therefore unchanged.

    Since a pyramid staircase can be approached from any side, the nose is
    added around all four outward-facing edges of every level.
    """
    if cfg.holes:
        raise ValueError(
            "MeshPyramidStairsWithNosingTerrainCfg does not support holes=True."
        )
    if cfg.nosing_depth <= 0.0:
        raise ValueError(f"nosing_depth must be positive, got {cfg.nosing_depth}.")
    if cfg.nosing_thickness <= 0.0:
        raise ValueError(
            f"nosing_thickness must be positive, got {cfg.nosing_thickness}."
        )
    if cfg.nosing_depth >= cfg.step_width:
        raise ValueError(
            f"nosing_depth ({cfg.nosing_depth}) must be smaller than step_width ({cfg.step_width})."
        )

    meshes, origin = pyramid_stairs_terrain(difficulty, cfg)
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )

    # At curriculum difficulty zero the base stair generator produces a flat
    # surface. Do not create a lip that would introduce height on that surface.
    if step_height <= 0.0:
        return meshes, origin

    terrain_size = (
        cfg.size[0] - 2.0 * cfg.border_width,
        cfg.size[1] - 2.0 * cfg.border_width,
    )
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1])
    num_steps_x = (cfg.size[0] - 2.0 * cfg.border_width - cfg.platform_width) // (
        2.0 * cfg.step_width
    ) + 1
    num_steps_y = (cfg.size[1] - 2.0 * cfg.border_width - cfg.platform_width) // (
        2.0 * cfg.step_width
    ) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
    nose_thickness = min(cfg.nosing_thickness, step_height)

    for level in range(num_steps + 1):
        footprint_x = terrain_size[0] - 2.0 * level * cfg.step_width
        footprint_y = terrain_size[1] - 2.0 * level * cfg.step_width
        edge_x = 0.5 * footprint_x
        edge_y = 0.5 * footprint_y
        nose_z = (level + 1) * step_height - 0.5 * nose_thickness

        # +/-x strips project toward the lower level. The +/-y strips extend
        # through the four corners so that the nose is continuous.
        x_strip_dims = (cfg.nosing_depth, footprint_y, nose_thickness)
        y_strip_dims = (
            footprint_x + 2.0 * cfg.nosing_depth,
            cfg.nosing_depth,
            nose_thickness,
        )
        nose_specs = (
            (
                x_strip_dims,
                (
                    terrain_center[0] + edge_x + 0.5 * cfg.nosing_depth,
                    terrain_center[1],
                    nose_z,
                ),
            ),
            (
                x_strip_dims,
                (
                    terrain_center[0] - edge_x - 0.5 * cfg.nosing_depth,
                    terrain_center[1],
                    nose_z,
                ),
            ),
            (
                y_strip_dims,
                (
                    terrain_center[0],
                    terrain_center[1] + edge_y + 0.5 * cfg.nosing_depth,
                    nose_z,
                ),
            ),
            (
                y_strip_dims,
                (
                    terrain_center[0],
                    terrain_center[1] - edge_y - 0.5 * cfg.nosing_depth,
                    nose_z,
                ),
            ),
        )
        for dimensions, position in nose_specs:
            nose_mesh = trimesh.creation.box(
                dimensions,
                trimesh.transformations.translation_matrix(position),
            )
            nose_mesh.metadata.update({"kind": "stair_nosing", "level": level})
            meshes.append(nose_mesh)

    return meshes, origin


@configclass
class MeshPyramidStairsWithNosingTerrainCfg(terrain_gen.MeshPyramidStairsTerrainCfg):
    """Configuration for pyramid stairs with a projecting stair nose."""

    function = pyramid_stairs_with_nosing_terrain

    nosing_depth: float = 0.01
    """Horizontal projection toward the lower tread (in m)."""

    nosing_thickness: float = 0.01
    """Vertical thickness measured down from the unchanged step top (in m)."""
