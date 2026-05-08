import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parent / "legged_gym" / "utils" / "warp_render_v3.py"
MODULE_SPEC = importlib.util.spec_from_file_location("warp_render_v3_under_test", MODULE_PATH)
warp_render_v3 = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(warp_render_v3)


def test_depth_renderer_initializes_warp_before_from_torch(monkeypatch):
    init_calls = []

    def fake_init():
        init_calls.append("init")

    def fake_from_torch(tensor, dtype=None):
        assert init_calls == ["init"]
        return ("fake-warp-array", tensor.shape, dtype)

    monkeypatch.setattr(warp_render_v3.wp, "init", fake_init)
    monkeypatch.setattr(warp_render_v3.wp, "from_torch", fake_from_torch)

    renderer = warp_render_v3.DepthRendererWarp(
        image_params=[2, 3],
        cam2base_xyz=torch.zeros((1, 3), dtype=torch.float32),
        cam2base_euler=torch.zeros((1, 3), dtype=torch.float32),
        fovy=torch.full((1, 1), 90.0, dtype=torch.float32),
        num_envs=1,
        device="cpu",
    )

    assert init_calls == ["init"]
    assert renderer.fovy_dist_offset[0] == "fake-warp-array"
