"""Standalone verification + benchmark for the per-env full-body BVH path.

This script does NOT require IsaacGym.  It directly imports
`legged_gym.utils.robot_geom`, `legged_gym.utils.h1_geom`, and
`legged_gym.utils.warp_render_v3` and exercises the renderer with a fake
multi-env world:

  * Build the H1 link template from `H1_LINK_GEOMS`.
  * Construct `DepthRendererWarp` with N envs and a flat terrain.
  * Drive the per-env "rigid body states" by hand to put a couple of envs in
    diagnostic poses.
  * Verify:
      1. Each env sees its own body (self-occlusion present).
      2. Cross-env isolation: env i's body is NOT visible in env j's depth
         even when geometrically inside env j's camera FOV.
  * Benchmark `update_robot_meshes` + `render_depth` for a set of env counts.
  * Save a per-env depth PNG to ``/tmp/h1_self_occ_envXX.png`` for visual
    inspection.

Usage
-----
    python legged_gym/scripts/verify_self_occlusion_bvh.py [--envs 256 1024 4096]

Notes
-----
The script imports the warp / robot_geom modules directly (bypassing
``legged_gym/__init__.py``) so that it can run without isaacgym installed.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from typing import List

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Importing without going through legged_gym/__init__.py (which pulls
# isaacgym).  Instead we load each module by file path.
# ---------------------------------------------------------------------------
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UTILS_DIR = os.path.join(WORKSPACE, "legged_gym", "utils")


def _load_local(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# robot_geom must load BEFORE warp_render_v3 so it shows up in sys.modules
# under the short name; h1_geom imports `from .robot_geom import LinkGeom`
# so we wire it via a fake package.
import types
pkg = types.ModuleType("_legged_gym_utils")
pkg.__path__ = [UTILS_DIR]
sys.modules["_legged_gym_utils"] = pkg
rg = _load_local("_legged_gym_utils.robot_geom", os.path.join(UTILS_DIR, "robot_geom.py"))
warprender = _load_local("warp_render_v3", os.path.join(UTILS_DIR, "warp_render_v3.py"))
h1g = _load_local("_legged_gym_utils.h1_geom", os.path.join(UTILS_DIR, "h1_geom.py"))


def _quat_xyzw_identity(num_envs: int, device) -> torch.Tensor:
    q = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
    q[:, 3] = 1.0
    return q


def _quat_wxyz_identity(num_envs: int, device) -> torch.Tensor:
    q = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
    q[:, 0] = 1.0
    return q


def _flat_terrain(half_size: float = 50.0):
    """Two big triangles forming a flat XY ground plane at z=0 in *gym* frame."""
    v = np.array([
        [-half_size, -half_size, 0.0],
        [+half_size, -half_size, 0.0],
        [+half_size, +half_size, 0.0],
        [-half_size, +half_size, 0.0],
    ], dtype=np.float32)
    t = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return v, t


def _build_renderer(num_envs: int, image_hw=(48, 64), device="cuda:0"):
    H, W = image_hw
    # H1 default camera config (h1_loco_config.depth):
    #   position = (0.0576, 0.0175, 0.43) in pelvis frame
    #   y_angle ~= 45 deg downward, x/z = 0
    #   fovy = 79.3 deg
    cam_xyz = torch.tensor([[0.0576235, 0.01753, 0.42987]] * num_envs,
                           dtype=torch.float32)
    cam_euler_deg = torch.zeros(num_envs, 3, dtype=torch.float32)
    cam_euler_deg[:, 1] = 45.0
    cam_euler = torch.deg2rad(cam_euler_deg)
    fovy = torch.full((num_envs, 1), 79.3, dtype=torch.float32)

    far_clip = 2.0
    far_t = far_clip + 0.2
    renderer = warprender.DepthRendererWarp(
        image_params=[H, W],
        cam2base_xyz=cam_xyz,
        cam2base_euler=cam_euler,
        fovy=fovy,
        num_envs=num_envs,
        far_t=far_t,
        miss_t=far_t,
        device=device,
    )

    # Terrain: flat ground centered at origin.
    tv, tt = _flat_terrain(half_size=200.0)
    renderer.render_mesh(tv, tt)
    return renderer


def _build_h1_template():
    return rg.build_robot_template(h1g.H1_LINK_GEOMS)


def _attach_h1_robot_meshes(renderer, template, refit_stride=1):
    # `body_indices` here is a fake mapping: we pretend each link in
    # template.link_names is at gym body index `i`.  In a real run the env
    # produces this mapping by looking up the actual IsaacGym body_names.
    body_indices = np.arange(len(template.link_names), dtype=np.int32)
    renderer.init_robot_meshes(
        template_verts_local=template.verts_local,
        template_tris=template.tris,
        vert_to_link=template.vert_to_link,
        body_indices=body_indices,
        refit_stride=refit_stride,
    )
    return body_indices


def _set_static_h1_pose(num_envs: int, num_bodies: int, device,
                        env_origins: torch.Tensor) -> torch.Tensor:
    """Place each link of each env at its nominal pose (h1.xml defaults).

    For verification we don't need to honor every joint chain - we just need
    a recognisable pose where pelvis is upright and the arms / legs are at
    sensible positions.  Returns a (num_envs, num_bodies, 13) tensor.
    """
    # Nominal H1 link offsets in WORLD (pelvis frame) when standing with
    # default joint angles; computed by walking the kinematic chain in
    # h1.xml.
    # Order MUST match template.link_names below.
    nominal = {
        "pelvis":                       (0.000,  0.000,  0.000),
        "torso_link":                   (0.000,  0.000,  0.000),
        "left_hip_pitch_link":          (0.039,  0.203, -0.174),
        "right_hip_pitch_link":         (0.039, -0.203, -0.174),
        "left_knee_link":               (0.039,  0.203, -0.574),
        "right_knee_link":              (0.039, -0.203, -0.574),
        "left_ankle_link":              (0.039,  0.203, -0.974),
        "right_ankle_link":             (0.039, -0.203, -0.974),
        # arm chain offsets (URDF: shoulder_pitch->roll->yaw->elbow are
        # all type="fixed" in H1; positions are accumulated transforms).
        "left_shoulder_pitch_link":     (0.006,  0.155,  0.430),
        "right_shoulder_pitch_link":    (0.006, -0.155,  0.430),
        # elbow_link world pos when the chain is at default angles:
        # shoulder_yaw_link drops 0.134 below shoulder_pitch, then elbow_link
        # is offset by (0.0185, 0, -0.198) from shoulder_yaw_link.  Sum:
        # x = 0.006 + 0.0185 ~ 0.025; y = ±0.214; z = 0.43 - 0.134 - 0.198 ~ 0.10
        "left_elbow_link":              (0.025,  0.214,  0.098),
        "right_elbow_link":             (0.025, -0.214,  0.098),
    }

    states = torch.zeros(num_envs, num_bodies, 13, dtype=torch.float32,
                         device=device)
    # identity quat (xyzw)
    states[:, :, 6] = 1.0

    # H1 stands upright with pelvis at base_init z = 1.0
    pelvis_z = 1.0

    for body_idx, link_name in enumerate(_link_order):
        pos = nominal.get(link_name, (0.0, 0.0, 0.0))
        states[:, body_idx, 0] = env_origins[:, 0] + pos[0]
        states[:, body_idx, 1] = env_origins[:, 1] + pos[1]
        states[:, body_idx, 2] = env_origins[:, 2] + pelvis_z + pos[2]
    return states


_link_order: List[str] = []   # set inside main()


def _save_depth_png(depth_torch: torch.Tensor, path: str):
    img = depth_torch.cpu().numpy()
    # depth is in [0, far_t]; remap to [0, 255]
    img = np.clip(img / (img.max() + 1e-6), 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    cv2.imwrite(path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", type=int,
                        default=[256, 1024, 4096],
                        help="env counts to benchmark")
    parser.add_argument("--refit-strides", nargs="+", type=int,
                        default=[1],
                        help="refit_stride values to sweep per env-count")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-bench", action="store_true",
                        help="skip the multi-env benchmark loop")
    args = parser.parse_args()

    import warp as wp
    wp.init()

    print("=" * 70)
    print("Per-env full-body BVH self-occlusion verification")
    print("=" * 70)

    # ---- 1. Correctness: a small num_envs run with diagnostic poses ----
    print("\n[1] Correctness check on 4 envs")
    NUM_ENVS = 4
    template = _build_h1_template()
    print(f"    H1 template: {template.num_verts} verts, "
          f"{template.num_tris} tris, links={template.link_names}")

    global _link_order
    _link_order = list(template.link_names)
    num_bodies = len(_link_order)

    renderer = _build_renderer(NUM_ENVS, device=args.device)
    body_indices = _attach_h1_robot_meshes(renderer, template)

    # space envs out far enough to *guarantee* no leakage even without the
    # per-env isolation — this is purely so the visual references are clean.
    spacing = 10.0
    env_origins = torch.tensor([
        [0.0, 0.0, 0.0],
        [spacing, 0.0, 0.0],
        [2 * spacing, 0.0, 0.0],
        [3 * spacing, 0.0, 0.0],
    ], dtype=torch.float32, device=args.device)

    states = _set_static_h1_pose(NUM_ENVS, num_bodies, args.device, env_origins)
    renderer.update_robot_meshes(states)

    # camera attached to pelvis: base_pos = pelvis pos, base_quat = pelvis quat
    base_pos = states[:, 0, 0:3]                  # pelvis world pos
    base_quat = states[:, 0, 3:7]                 # pelvis world quat (xyzw)
    depth = renderer.render_depth(base_pos, base_quat)
    print(f"    depth shape: {tuple(depth.shape)}, "
          f"min={float(depth.min()):.3f}, "
          f"max={float(depth.max()):.3f}, "
          f"mean={float(depth.mean()):.3f}")
    print(f"    fraction of pixels < 1.5m (near-field): "
          f"{float((depth < 1.5).float().mean()):.3f}")

    out_dir = "/tmp"
    for i in range(min(NUM_ENVS, 4)):
        path = os.path.join(out_dir, f"h1_self_occ_env{i}.png")
        _save_depth_png(depth[i], path)
        print(f"    saved {path}  (max={float(depth[i].max()):.3f})")

    # ---- 2. Cross-env isolation test ----
    print("\n[2] Cross-env isolation test")
    # Move env 0's pelvis (and thus all its links via static pose) into env 1's
    # FOV.  Without per-env BVH isolation, env 1 would see env 0's body and
    # the depth would change; with isolation it MUST stay the same.
    # We'll first record env 1's depth, then move env 0's body to within env
    # 1's FOV, then re-render and confirm env 1's depth is unchanged.
    env1_depth_before = depth[1].clone()

    # move env 0's pelvis next to env 1
    new_origins = env_origins.clone()
    new_origins[0] = env_origins[1] + torch.tensor([1.0, 0.0, 0.0],
                                                   device=args.device)
    states = _set_static_h1_pose(NUM_ENVS, num_bodies, args.device, new_origins)
    renderer.update_robot_meshes(states)

    base_pos2 = states[:, 0, 0:3]
    base_quat2 = states[:, 0, 3:7]
    depth2 = renderer.render_depth(base_pos2, base_quat2)
    env1_depth_after = depth2[1]

    diff = (env1_depth_after - env1_depth_before).abs().max()
    print(f"    env 1 depth max-abs change after moving env 0 next to it: "
          f"{float(diff):.6f} m")
    if float(diff) < 1e-4:
        print("    PASS: env 1 was unaffected by env 0's body (per-env BVH "
              "isolation works).")
    else:
        print("    FAIL: env 1 saw a change > 1e-4 m; cross-env leak!")

    # ---- 3. Benchmark scaling ----
    if not args.no_bench:
        print("\n[3] Benchmark scaling (update + render)")
        print(f"    target: well under 100 ms = 10 Hz depth refresh budget")
        for n in args.envs:
            for stride in args.refit_strides:
                renderer = _build_renderer(n, device=args.device)
                _attach_h1_robot_meshes(renderer, template, refit_stride=stride)

                # spread envs in a grid 4 m apart so cameras don't see
                # neighbors' terrain features (all is flat anyway).
                cols = int(np.ceil(np.sqrt(n)))
                origins = torch.zeros(n, 3, dtype=torch.float32,
                                      device=args.device)
                origins[:, 0] = (torch.arange(n) % cols) * 4.0
                origins[:, 1] = (torch.arange(n) // cols) * 4.0

                states = _set_static_h1_pose(n, num_bodies, args.device, origins)
                base_pos = states[:, 0, 0:3]
                base_quat = states[:, 0, 3:7]

                # warmup; ensure refit_stride doesn't skip the first update
                for _ in range(stride * 2):
                    renderer.update_robot_meshes(states)
                    _ = renderer.render_depth(base_pos, base_quat)
                torch.cuda.synchronize()

                # n_iter must be at least 2*stride so the amortized average
                # includes at least one full refit cycle.
                n_iter = max(5, stride * 3)
                t_upd = 0.0
                t_ren = 0.0
                for _ in range(n_iter):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    renderer.update_robot_meshes(states)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    _ = renderer.render_depth(base_pos, base_quat)
                    torch.cuda.synchronize()
                    t2 = time.time()
                    t_upd += (t1 - t0)
                    t_ren += (t2 - t1)
                # average over n_iter calls; on stride > 1, only every Nth
                # call actually refits, so the average update cost is the
                # amortized cost across N depth ticks.
                print(f"    num_envs={n:5d}  refit_stride={stride}  "
                      f"update={t_upd*1000/n_iter:7.2f} ms  "
                      f"render={t_ren*1000/n_iter:7.2f} ms  "
                      f"total={(t_upd+t_ren)*1000/n_iter:7.2f} ms / depth tick")

    print("\nDone.")


if __name__ == "__main__":
    main()
