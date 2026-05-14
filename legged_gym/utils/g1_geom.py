"""Simplified per-link geometry template for Unitree G1.

This file provides a coarse approximation of the G1 body using box/capsule
primitives in each link's local frame. The template is used for per-env BVH
self-occlusion in the warp renderer.
"""
from typing import List

from .robot_geom import LinkGeom


# NOTE: Dimensions are approximate and tuned for visibility, not exact fit.
G1_LINK_GEOMS: List[LinkGeom] = [
    # ----- thighs -----
    LinkGeom(
        link_name="left_hip_pitch_link",
        primitive="capsule",
        params=(0.06, 0.38),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.19),
    ),
    LinkGeom(
        link_name="right_hip_pitch_link",
        primitive="capsule",
        params=(0.06, 0.38),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.19),
    ),

    # ----- shins -----
    LinkGeom(
        link_name="left_knee_link",
        primitive="capsule",
        params=(0.055, 0.38),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.19),
    ),
    LinkGeom(
        link_name="right_knee_link",
        primitive="capsule",
        params=(0.055, 0.38),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.19),
    ),

    # ----- feet -----
    LinkGeom(
        link_name="left_ankle_pitch_link",
        primitive="box",
        params=(0.20, 0.08, 0.05),
        offset_xyz=(0.06, 0.0, -0.04),
    ),
    LinkGeom(
        link_name="right_ankle_pitch_link",
        primitive="box",
        params=(0.20, 0.08, 0.05),
        offset_xyz=(0.06, 0.0, -0.04),
    ),

    # ----- upper arms -----
    LinkGeom(
        link_name="left_shoulder_pitch_link",
        primitive="capsule",
        params=(0.045, 0.28),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.14),
    ),
    LinkGeom(
        link_name="right_shoulder_pitch_link",
        primitive="capsule",
        params=(0.045, 0.28),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.14),
    ),

    # ----- forearms -----
    LinkGeom(
        link_name="left_elbow_link",
        primitive="capsule",
        params=(0.04, 0.22),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.11),
    ),
    LinkGeom(
        link_name="right_elbow_link",
        primitive="capsule",
        params=(0.04, 0.22),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.11),
    ),
]


__all__ = ["G1_LINK_GEOMS"]
