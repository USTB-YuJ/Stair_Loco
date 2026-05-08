"""Simplified per-link geometry template for the Unitree H1.

This file describes a coarse-but-reasonable approximation of every H1 rigid
body that may be visible from the pelvis-mounted depth camera.  Each entry is
a single primitive (box / capsule) whose pose is given in the *link-local*
frame (i.e. before applying the link's world pose).

We deliberately cover the **whole body** (legs + torso + arms), not just the
links that are typically in FOV.  Reason: any unmodeled body part that ever
enters the camera's view at deployment time becomes an OOD feature for the
policy.  Even a small near-camera spike can change tens of pixels and trigger
unexpected behaviour.  The cost is paid at training time (BVH refit) and
amortized via ``cfg.depth.refit_stride``.

Link offsets and axes are derived from
``resources/robots/h1/urdf/h1.urdf`` and ``resources/robots/h1/h1.xml``.
The H1's torso-and-arms are all glued together by ``type="fixed"`` joints,
so their relative transforms never change.  We still attach each capsule to
its *own* rigid body (rather than baking everything into the pelvis) so that
the renderer stays correct in case any future config un-fixes those joints.

Coordinate conventions (link-local, identical to MuJoCo / URDF):
  +X : forward (toward the toes)
  +Y : robot's left
  +Z : up

Capsules are built along their +Z axis by default; the leg / arm capsules
extend in the -Z direction from the joint origin (hip / knee / shoulder /
elbow), so we centre them at ``(0, 0, -length/2)``.
"""
from typing import List

from .robot_geom import LinkGeom


# ---------------------------------------------------------------------------
# Per-link primitives.
# ---------------------------------------------------------------------------
# Triangle counts (capsule defaults n_seg=6, n_cap=1 -> 48 tri each;
# box -> 12 tri):
#   pelvis box           :  12
#   torso  box           :  12
#   2 thigh   capsules   :  96
#   2 shin    capsules   :  96
#   2 foot    boxes      :  24
#   2 upper-arm capsules :  96   (shoulder_pitch_link -> elbow region)
#   2 fore-arm  capsules :  96   (elbow_link -> hand)
#   ----------------------
#                  total : 432 tri / env
H1_LINK_GEOMS: List[LinkGeom] = [
    # ----- pelvis / torso (fixed bounding boxes for the trunk) -----
    LinkGeom(
        link_name="pelvis",
        primitive="box",
        params=(0.22, 0.30, 0.20),                   # forward x lateral x vertical
        offset_xyz=(0.0, 0.0, -0.05),                # pelvis frame origin sits near hip joint level
    ),
    # The torso box covers the chest region.  The camera (D435i) is now on the
    # head at z=0.693 in pelvis frame, so the torso box must stay below that.
    # We split the upper body into a chest box and a head box.
    LinkGeom(
        link_name="torso_link",
        primitive="box",
        params=(0.16, 0.36, 0.45),                   # forward x lateral x vertical
        offset_xyz=(-0.03, 0.0, 0.22),               # chest centre at z=0.22, top at z=0.445
    ),
    # Head region: the camera sits at (0.108, 0.018, 0.693). The head itself
    # extends from ~z=0.50 to ~z=0.72 and is narrower than the torso. We
    # place the box *behind* the camera to avoid enclosing it.
    LinkGeom(
        link_name="torso_link",
        primitive="box",
        params=(0.12, 0.18, 0.22),                   # forward x lateral x vertical
        offset_xyz=(-0.04, 0.0, 0.60),               # head centre at z=0.60, top at z=0.71
    ),

    # ----- thighs (hip_pitch_link -> knee_link, length 0.4) -----
    LinkGeom(
        link_name="left_hip_pitch_link",
        primitive="capsule",
        params=(0.07, 0.40),                         # radius, length (cylindrical part)
        axis="z",
        offset_xyz=(0.0, 0.0, -0.20),                # extend downward 0.4 from hip
    ),
    LinkGeom(
        link_name="right_hip_pitch_link",
        primitive="capsule",
        params=(0.07, 0.40),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.20),
    ),

    # ----- shins (knee_link -> ankle_link, length 0.4) -----
    LinkGeom(
        link_name="left_knee_link",
        primitive="capsule",
        params=(0.06, 0.40),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.20),
    ),
    LinkGeom(
        link_name="right_knee_link",
        primitive="capsule",
        params=(0.06, 0.40),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.20),
    ),

    # ----- feet (ankle_link, foot is forward-extended box) -----
    # ankle_link inertial is at (0.0486, 0, -0.0456); we approximate the foot as
    # a long flat box centred under and slightly in front of the ankle joint.
    LinkGeom(
        link_name="left_ankle_link",
        primitive="box",
        params=(0.22, 0.08, 0.05),
        offset_xyz=(0.06, 0.0, -0.04),
    ),
    LinkGeom(
        link_name="right_ankle_link",
        primitive="box",
        params=(0.22, 0.08, 0.05),
        offset_xyz=(0.06, 0.0, -0.04),
    ),

    # ----- upper arms (shoulder_pitch_link -> ~elbow) -----
    # The arm chain is rigid (shoulder_pitch/roll/yaw and elbow are all
    # type="fixed" in the URDF), so a single capsule per side at the
    # shoulder_pitch_link is enough to follow it.  We use TWO capsules per
    # arm (upper + forearm) so that close-up shots of either segment look
    # like an actual arm, not a single thick rod.
    #
    # In shoulder_pitch_link frame, the chain drops ~0.32 m to the elbow
    # along (mostly) -Z.  The capsule covers shoulder pitch joint down to
    # roughly the elbow.
    LinkGeom(
        link_name="left_shoulder_pitch_link",
        primitive="capsule",
        params=(0.05, 0.32),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.16),
    ),
    LinkGeom(
        link_name="right_shoulder_pitch_link",
        primitive="capsule",
        params=(0.05, 0.32),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.16),
    ),

    # ----- forearm + hand (elbow_link -> end of arm) -----
    # Elbow link frame origin sits at the elbow joint; the forearm + hand
    # extend ~0.25 m further along -Z.
    LinkGeom(
        link_name="left_elbow_link",
        primitive="capsule",
        params=(0.045, 0.25),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.125),
    ),
    LinkGeom(
        link_name="right_elbow_link",
        primitive="capsule",
        params=(0.045, 0.25),
        axis="z",
        offset_xyz=(0.0, 0.0, -0.125),
    ),
]


# Backwards-compat alias: the arm capsules used to live in a separate optional
# list.  They are now part of the default template; this name is kept so any
# downstream code that did ``H1_LINK_GEOMS + H1_LINK_GEOMS_ARMS`` still works
# (the empty list makes that a no-op).
H1_LINK_GEOMS_ARMS: List[LinkGeom] = []


__all__ = ["H1_LINK_GEOMS", "H1_LINK_GEOMS_ARMS"]
