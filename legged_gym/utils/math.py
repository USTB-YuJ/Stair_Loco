import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

def quat_inv(q):
    return q * __import__("torch").tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)


def quat_mul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return __import__("torch").stack([w, x, y, z], dim=-1)


def convert_camera_frame_orientation_convention(quat, origin="opengl", target="world"):
    if origin == target:
        return quat
    q_gl_to_world = __import__("torch").tensor([0.5, 0.5, -0.5, -0.5], device=quat.device, dtype=quat.dtype)
    q_world_to_gl = __import__("torch").tensor([0.5, -0.5, 0.5, 0.5], device=quat.device, dtype=quat.dtype)
    if origin == "opengl" and target in ("world", "ros"):
        return quat_mul(q_gl_to_world, quat)
    elif origin in ("world", "ros") and target == "opengl":
        return quat_mul(q_world_to_gl, quat)
    else:
        raise ValueError(f"Unsupported convention conversion: {origin} -> {target}")
