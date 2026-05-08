import math

FX = 384.77294921875
FY = 384.77294921875
CX = 324.17236328125
CY = 236.48226928710938
FULL_H = 480
FULL_W = 640
CROP_H = 48
CROP_W = 64


def _warp_ray_direction(u, v):
    raw = (1.0, -(u - CX) / FX, -(v - CY) / FY)
    norm = math.sqrt(sum(x * x for x in raw))
    return tuple(x / norm for x in raw)


def _bottom_center_crop_window(src_h, src_w, target_h, target_w, bottom_margin=0):
    top = src_h - bottom_margin - target_h
    left = (src_w - target_w) // 2
    return top, top + target_h, left, left + target_w


def test_warp_ray_uses_configured_pinhole_intrinsics():
    center = _warp_ray_direction(CX, CY)
    assert math.isclose(center[1] / center[0], 0.0, abs_tol=1e-12)
    assert math.isclose(center[2] / center[0], 0.0, abs_tol=1e-12)

    u, v = 320.0, 432.0
    ray = _warp_ray_direction(u, v)
    assert math.isclose(ray[1] / ray[0], -(u - CX) / FX, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ray[2] / ray[0], -(v - CY) / FY, rel_tol=0.0, abs_tol=1e-12)


def test_h1_depth_bottom_center_crop_window():
    assert _bottom_center_crop_window(FULL_H, FULL_W, CROP_H, CROP_W) == (432, 480, 288, 352)


if __name__ == "__main__":
    test_warp_ray_uses_configured_pinhole_intrinsics()
    test_h1_depth_bottom_center_crop_window()
    print("depth camera math checks passed")
