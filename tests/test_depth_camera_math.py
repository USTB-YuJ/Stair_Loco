import math
import importlib.util
from pathlib import Path

FX = 384.77294921875
FY = 384.77294921875
CX = 324.17236328125
CY = 236.48226928710938
FULL_H = 480
FULL_W = 640
CROP_H = 48
CROP_W = 64


def _load_depth_roi_module():
    module_path = Path(__file__).resolve().parents[1] / "legged_gym" / "utils" / "depth_roi.py"
    spec = importlib.util.spec_from_file_location("depth_roi_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _warp_ray_direction(u, v):
    raw = (1.0, -(u - CX) / FX, -(v - CY) / FY)
    norm = math.sqrt(sum(x * x for x in raw))
    return tuple(x / norm for x in raw)


def _bottom_center_crop_window(src_h, src_w, target_h, target_w, bottom_margin=0):
    top = src_h - bottom_margin - target_h
    left = (src_w - target_w) // 2
    return top, top + target_h, left, left + target_w


def test_h1_depth_roi_crop_window_handles_zero_bottom_crop():
    depth_roi = _load_depth_roi_module()

    assert depth_roi.crop_window_from_pixels((96, 128), [32, 32, 32, 0]) == (32, 96, 32, 96)


def test_h1_depth_roi_crop_window_rejects_empty_crop():
    depth_roi = _load_depth_roi_module()

    try:
        depth_roi.crop_window_from_pixels((96, 128), [64, 32, 64, 0])
    except ValueError as exc:
        assert "Invalid crop" in str(exc)
    else:
        raise AssertionError("Expected invalid full-width crop to raise ValueError")


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
