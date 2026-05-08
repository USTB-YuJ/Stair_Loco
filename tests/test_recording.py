import importlib.util
from pathlib import Path
import subprocess

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "legged_gym" / "utils" / "recording.py"
MODULE_SPEC = importlib.util.spec_from_file_location("recording_under_test", MODULE_PATH)
recording = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(recording)

PYTHON38 = Path("/root/gpufree-data/conda_space/envs/more/bin/python3.8")


def test_prepare_recording_returns_disabled_config_when_not_enabled(tmp_path):
    info = recording.prepare_recording(
        enabled=False,
        record_dir=tmp_path,
        record_name="demo",
    )

    assert info.enabled is False
    assert info.frames_dir is None
    assert info.frame_pattern is None
    assert info.video_path is None


def test_recording_module_imports_under_python38():
    result = subprocess.run(
        [
            str(PYTHON38),
            "-c",
            (
                "import importlib.util; "
                "spec = importlib.util.spec_from_file_location("
                "'recording_under_test', r'/root/gpufree-data/workspace/more/legged_gym/utils/recording.py'); "
                "module = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(module)"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_prepare_recording_creates_frame_pattern(tmp_path):
    info = recording.prepare_recording(
        enabled=True,
        record_dir=tmp_path,
        record_name="demo",
    )

    assert info.enabled is True
    assert info.frames_dir == (tmp_path / "demo_frames").resolve()
    assert info.frames_dir.is_dir()
    assert info.frame_pattern == str((tmp_path / "demo_frames" / "frame_%06d.png").resolve())
    assert info.video_path == (tmp_path / "demo.mp4").resolve()


def test_finalize_recording_runs_ffmpeg_and_cleans_frames(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))
        (tmp_path / "demo.mp4").write_bytes(b"mp4")

    info = recording.prepare_recording(
        enabled=True,
        record_dir=tmp_path,
        record_name="demo",
    )
    (info.frames_dir / "frame_000001.png").write_bytes(b"png")

    monkeypatch.setattr(recording.subprocess, "run", fake_run)

    video_path = recording.finalize_recording(info, fps=30, keep_frames=False)

    assert calls
    ffmpeg_cmd, check_value = calls[0]
    assert check_value is True
    assert ffmpeg_cmd[:4] == ["ffmpeg", "-y", "-framerate", "30"]
    assert ffmpeg_cmd[-1] == str((tmp_path / "demo.mp4").resolve())
    assert video_path == (tmp_path / "demo.mp4").resolve()
    assert not info.frames_dir.exists()


def test_finalize_recording_raises_when_enabled_but_no_frames_exist(tmp_path):
    info = recording.prepare_recording(
        enabled=True,
        record_dir=tmp_path,
        record_name="demo",
    )

    with pytest.raises(RuntimeError, match="No recorded frames"):
        recording.finalize_recording(info, fps=30, keep_frames=False)


def test_finalize_recording_if_available_returns_none_when_no_frames_exist(tmp_path):
    info = recording.prepare_recording(
        enabled=True,
        record_dir=tmp_path,
        record_name="demo",
    )

    assert recording.finalize_recording_if_available(info, fps=30, keep_frames=False) is None


def test_finalize_recording_if_available_exports_video_when_frames_exist(tmp_path, monkeypatch):
    info = recording.prepare_recording(
        enabled=True,
        record_dir=tmp_path,
        record_name="demo",
    )
    (info.frames_dir / "frame_000001.png").write_bytes(b"png")

    expected_video = (tmp_path / "demo.mp4").resolve()
    calls = []

    def fake_finalize(recording_info, fps, keep_frames):
        calls.append((recording_info, fps, keep_frames))
        return expected_video

    monkeypatch.setattr(recording, "finalize_recording", fake_finalize)

    video_path = recording.finalize_recording_if_available(info, fps=24, keep_frames=True)

    assert video_path == expected_video
    assert calls == [(info, 24, True)]


def test_finalize_recording_falls_back_to_imageio_when_ffmpeg_fails(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check):
        raise recording.subprocess.CalledProcessError(returncode=127, cmd=cmd)

    def fake_encode(frame_files, video_path, fps):
        calls.append((frame_files, video_path, fps))
        video_path.write_bytes(b"mp4")

    info = recording.prepare_recording(
        enabled=True,
        record_dir=tmp_path,
        record_name="demo",
    )
    frame_path = info.frames_dir / "frame_000001.png"
    frame_path.write_bytes(b"png")

    monkeypatch.setattr(recording.subprocess, "run", fake_run)
    monkeypatch.setattr(recording, "_encode_video_with_imageio", fake_encode, raising=False)

    video_path = recording.finalize_recording(info, fps=24, keep_frames=False)

    assert calls == [([frame_path], (tmp_path / "demo.mp4").resolve(), 24)]
    assert video_path == (tmp_path / "demo.mp4").resolve()
    assert not info.frames_dir.exists()


def test_finalize_recording_falls_back_to_cv2_when_imageio_backend_missing(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check):
        raise recording.subprocess.CalledProcessError(returncode=127, cmd=cmd)

    def fake_imageio(frame_files, video_path, fps):
        raise ValueError("no imageio backend")

    def fake_cv2(frame_files, video_path, fps):
        calls.append((frame_files, video_path, fps))
        video_path.write_bytes(b"mp4")

    info = recording.prepare_recording(
        enabled=True,
        record_dir=tmp_path,
        record_name="demo",
    )
    frame_path = info.frames_dir / "frame_000001.png"
    frame_path.write_bytes(b"png")

    monkeypatch.setattr(recording.subprocess, "run", fake_run)
    monkeypatch.setattr(recording, "_encode_video_with_imageio", fake_imageio, raising=False)
    monkeypatch.setattr(recording, "_encode_video_with_cv2", fake_cv2, raising=False)

    video_path = recording.finalize_recording(info, fps=24, keep_frames=False)

    assert calls == [([frame_path], (tmp_path / "demo.mp4").resolve(), 24)]
    assert video_path == (tmp_path / "demo.mp4").resolve()
    assert not info.frames_dir.exists()
