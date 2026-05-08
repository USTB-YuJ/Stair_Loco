"""Simplified per-link geometry primitives for the per-env warp BVH.

Each robot link is approximated by one (or a small number of) box / capsule
primitives expressed in the link's local frame.  At runtime, we transform the
link-local vertex template by the link's world pose (`rigid_body_states`) and
write it into the per-env warp mesh, which the depth renderer ray-casts
against to obtain self-occlusion.

Conventions
-----------
- All angles are in radians.
- ``offset_xyz`` / ``offset_rpy`` are the rigid transform of the primitive in
  the link frame (i.e. apply offset *before* the link's world pose).
- Capsules are oriented along their local +Z axis by default; ``axis`` can
  rotate them onto X/Y/Z by 0 / 90 / 90 degrees via ``offset_rpy`` in the
  caller, or you can use ``axis="x"|"y"|"z"`` in :class:`LinkGeom` and we'll
  handle the rotation here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import transforms3d as t3d


@dataclass
class LinkGeom:
    """A single primitive attached to one rigid body.

    Multiple ``LinkGeom`` entries may share the same ``link_name`` (e.g. the
    pelvis can carry both a torso box and a small pelvis box).
    """

    link_name: str
    primitive: str            # "box" | "capsule"
    params: Tuple[float, ...] # box: (sx, sy, sz); capsule: (radius, length)
    offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: str = "z"           # capsule cylinder axis: "x" | "y" | "z"


# ---------------------------------------------------------------------------
# Primitive mesh builders (return numpy arrays in float32 / int32)
# ---------------------------------------------------------------------------
def build_box_mesh(sx: float, sy: float, sz: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box centred at origin.

    Returns ``(verts[8, 3], tris[12, 3])``.  Triangle winding is outward.
    """
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    verts = np.array([
        [-hx, -hy, -hz],   # 0
        [+hx, -hy, -hz],   # 1
        [+hx, +hy, -hz],   # 2
        [-hx, +hy, -hz],   # 3
        [-hx, -hy, +hz],   # 4
        [+hx, -hy, +hz],   # 5
        [+hx, +hy, +hz],   # 6
        [-hx, +hy, +hz],   # 7
    ], dtype=np.float32)
    tris = np.array([
        [0, 2, 1], [0, 3, 2],     # -Z face
        [4, 5, 6], [4, 6, 7],     # +Z face
        [0, 1, 5], [0, 5, 4],     # -Y face
        [3, 6, 2], [3, 7, 6],     # +Y face
        [0, 4, 7], [0, 7, 3],     # -X face
        [1, 2, 6], [1, 6, 5],     # +X face
    ], dtype=np.int32)
    return verts, tris


def build_capsule_mesh(radius: float, length: float, axis: str = "z",
                       n_seg: int = 6, n_cap: int = 1
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Capsule (cylinder + two hemispheres) along the given axis.

    Defaults (n_seg=6, n_cap=1) produce a ~48-triangle approximation, which
    keeps the warp BVH refit cost low for thousands of parallel environments
    while still being a recognisable cylinder shape.  Bump up the sampling
    when you need a tighter fit.

    Parameters
    ----------
    radius : float
        Cylinder + hemisphere radius.
    length : float
        Cylinder length (axial distance between the two hemisphere centres).
    axis : {"x", "y", "z"}
        Local axis along which the capsule is laid out.
    n_seg : int
        Number of radial segments (around the cylinder).
    n_cap : int
        Number of latitude bands per hemisphere (excluding the pole).

    Returns
    -------
    verts : np.ndarray, shape (V, 3), float32
    tris  : np.ndarray, shape (T, 3), int32
    """
    half = length * 0.5
    rings: List[np.ndarray] = []          # each ring: (n_seg, 3)

    # ---- bottom hemisphere ----
    # latitudes from -pi/2 (bottom pole) up toward equator
    for i in range(1, n_cap + 1):
        phi = -np.pi / 2 + (np.pi / 2) * (i / (n_cap + 1))
        z_off = -half + radius * np.sin(phi)
        r_xy = radius * np.cos(phi)
        ring = _ring(r_xy, z_off, n_seg)
        rings.append(ring)

    # ---- cylinder side rings (equator at both ends) ----
    rings.append(_ring(radius, -half, n_seg))
    rings.append(_ring(radius, +half, n_seg))

    # ---- top hemisphere ----
    for i in range(n_cap, 0, -1):
        phi = +np.pi / 2 - (np.pi / 2) * (i / (n_cap + 1))
        z_off = +half + radius * np.sin(phi)
        r_xy = radius * np.cos(phi)
        rings.append(_ring(r_xy, z_off, n_seg))

    rings_np = np.concatenate(rings, axis=0)        # (n_rings * n_seg, 3)
    n_rings = len(rings)

    # poles
    bot_pole = np.array([[0.0, 0.0, -half - radius]], dtype=np.float32)
    top_pole = np.array([[0.0, 0.0, +half + radius]], dtype=np.float32)
    verts = np.concatenate([bot_pole, rings_np, top_pole], axis=0)

    bot_idx = 0
    top_idx = verts.shape[0] - 1

    tris: List[Tuple[int, int, int]] = []
    # bottom pole fan -> first ring
    for s in range(n_seg):
        a = 1 + s
        b = 1 + (s + 1) % n_seg
        tris.append((bot_idx, b, a))
    # quad strips between rings
    for r in range(n_rings - 1):
        base_a = 1 + r * n_seg
        base_b = 1 + (r + 1) * n_seg
        for s in range(n_seg):
            a0 = base_a + s
            a1 = base_a + (s + 1) % n_seg
            b0 = base_b + s
            b1 = base_b + (s + 1) % n_seg
            tris.append((a0, a1, b1))
            tris.append((a0, b1, b0))
    # top pole fan -> last ring
    last_ring_base = 1 + (n_rings - 1) * n_seg
    for s in range(n_seg):
        a = last_ring_base + s
        b = last_ring_base + (s + 1) % n_seg
        tris.append((top_idx, a, b))

    tris_np = np.array(tris, dtype=np.int32)

    # rotate to requested axis (capsule is built along Z by default)
    if axis == "x":
        R = t3d.euler.euler2mat(0.0, np.pi / 2, 0.0, "sxyz")
    elif axis == "y":
        R = t3d.euler.euler2mat(np.pi / 2, 0.0, 0.0, "sxyz")
    elif axis == "z":
        R = np.eye(3)
    else:
        raise ValueError(f"axis must be 'x'|'y'|'z', got {axis!r}")
    verts = verts @ R.T.astype(np.float32)

    return verts, tris_np


def _ring(radius: float, z: float, n_seg: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False, dtype=np.float32)
    return np.stack([radius * np.cos(angles),
                     radius * np.sin(angles),
                     np.full_like(angles, z)], axis=-1)


# ---------------------------------------------------------------------------
# Assembly: turn a list of LinkGeom into a single per-env vertex template
# ---------------------------------------------------------------------------
@dataclass
class RobotMeshTemplate:
    """Output of :func:`build_robot_template`.

    Holds the link-local vertex template that one env's BVH operates on.
    """

    verts_local: np.ndarray         # (V_total, 3)  in *link-local* frame
    tris: np.ndarray                # (T_total, 3)  index into verts_local
    vert_to_link: np.ndarray        # (V_total,)    -> index into link_names
    link_names: List[str]           # ordered, deduplicated

    @property
    def num_verts(self) -> int:
        return int(self.verts_local.shape[0])

    @property
    def num_tris(self) -> int:
        return int(self.tris.shape[0])


def build_robot_template(link_geoms: Sequence[LinkGeom]) -> RobotMeshTemplate:
    """Concatenate primitive meshes into one per-env template.

    For every primitive we:
      1. Build its mesh in primitive-local coords
      2. Apply the primitive's ``offset_rpy`` rotation
      3. Apply the primitive's ``offset_xyz`` translation (so verts now sit in
         the link-local frame)
      4. Append to the global verts, with a triangle index offset

    Each vertex is tagged with the index of the link it belongs to so the
    update kernel knows which body's world pose to apply.
    """
    all_verts: List[np.ndarray] = []
    all_tris: List[np.ndarray] = []
    vert_to_link: List[int] = []
    link_names: List[str] = []
    name_to_idx: dict[str, int] = {}

    v_offset = 0
    for geom in link_geoms:
        # build primitive-local mesh
        if geom.primitive == "box":
            sx, sy, sz = geom.params
            v, t = build_box_mesh(sx, sy, sz)
        elif geom.primitive == "capsule":
            radius, length = geom.params
            v, t = build_capsule_mesh(radius, length, axis=geom.axis)
        else:
            raise ValueError(f"Unknown primitive {geom.primitive!r}")

        # apply primitive -> link offset
        R = t3d.euler.euler2mat(geom.offset_rpy[0],
                                geom.offset_rpy[1],
                                geom.offset_rpy[2], "sxyz").astype(np.float32)
        v = v @ R.T + np.array(geom.offset_xyz, dtype=np.float32)

        # tag with link index
        if geom.link_name not in name_to_idx:
            name_to_idx[geom.link_name] = len(link_names)
            link_names.append(geom.link_name)
        link_idx = name_to_idx[geom.link_name]

        all_verts.append(v)
        all_tris.append(t + v_offset)
        vert_to_link.extend([link_idx] * v.shape[0])
        v_offset += v.shape[0]

    return RobotMeshTemplate(
        verts_local=np.concatenate(all_verts, axis=0).astype(np.float32),
        tris=np.concatenate(all_tris, axis=0).astype(np.int32),
        vert_to_link=np.array(vert_to_link, dtype=np.int32),
        link_names=link_names,
    )


__all__ = [
    "LinkGeom",
    "RobotMeshTemplate",
    "build_box_mesh",
    "build_capsule_mesh",
    "build_robot_template",
]
