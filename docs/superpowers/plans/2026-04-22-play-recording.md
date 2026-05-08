# Play Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a usable recording workflow to `legged_gym/scripts/play.py` that captures viewer frames and produces an mp4 file after playback.

**Architecture:** Keep Isaac Gym specific frame capture inside `play.py`, but move path preparation and ffmpeg orchestration into a pure Python helper module so the core recording behavior is easy to test without importing Isaac Gym. Extend CLI parsing in `helpers.py` so recording can be enabled and configured from the command line.

**Tech Stack:** Python, Isaac Gym viewer image capture, `subprocess`, `pathlib`, `pytest`

---

### Task 1: Add testable recording helpers

**Files:**
- Create: `legged_gym/utils/recording.py`
- Test: `tests/test_recording.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from legged_gym.utils.recording import finalize_recording, prepare_recording


def test_prepare_recording_creates_frame_pattern(tmp_path):
    info = prepare_recording(
        enabled=True,
        record_dir=tmp_path,
        record_name="demo",
    )

    assert info.enabled is True
    assert info.frames_dir == tmp_path / "demo_frames"
    assert info.frame_pattern == str((tmp_path / "demo_frames" / "frame_%06d.png").resolve())
    assert info.video_path == (tmp_path / "demo.mp4").resolve()


def test_finalize_recording_runs_ffmpeg_and_cleans_frames(tmp_path, monkeypatch):
    calls = []
    frames_dir = tmp_path / "demo_frames"
    frames_dir.mkdir()
    (frames_dir / "frame_000001.png").write_bytes(b"png")

    def fake_run(cmd, check):
        calls.append(cmd)
        (tmp_path / "demo.mp4").write_bytes(b"mp4")

    recording = prepare_recording(True, tmp_path, "demo")
    monkeypatch.setattr("legged_gym.utils.recording.subprocess.run", fake_run)

    finalize_recording(recording, fps=30, keep_frames=False)

    assert calls
    assert not frames_dir.exists()
    assert (tmp_path / "demo.mp4").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recording.py -v`
Expected: FAIL with import error because `legged_gym.utils.recording` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True)
class RecordingPaths:
    enabled: bool
    frames_dir: Path | None
    frame_pattern: str | None
    video_path: Path | None


def prepare_recording(enabled: bool, record_dir: str | Path, record_name: str) -> RecordingPaths:
    ...


def finalize_recording(recording: RecordingPaths, fps: int, keep_frames: bool) -> Path | None:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_recording.py -v`
Expected: PASS

### Task 2: Wire recording into play flow

**Files:**
- Modify: `legged_gym/scripts/play.py`
- Modify: `legged_gym/utils/helpers.py`
- Modify: `legged_gym/utils/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
def test_prepare_recording_returns_disabled_config_when_not_enabled(tmp_path):
    info = prepare_recording(False, tmp_path, "demo")

    assert info.enabled is False
    assert info.frames_dir is None
    assert info.video_path is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recording.py::test_prepare_recording_returns_disabled_config_when_not_enabled -v`
Expected: FAIL until disabled-recording behavior is implemented.

- [ ] **Step 3: Write minimal implementation**

```python
# helpers.py
{"name": "--record", "action": "store_true", "default": False, "help": "Capture viewer frames and export an mp4 after playback"},
{"name": "--record_dir", "type": str, "default": None, "help": "Directory used for temporary frames and output video"},
{"name": "--record_name", "type": str, "default": "play", "help": "Base file name for recorded video"},
{"name": "--record_fps", "type": int, "default": 50, "help": "Output mp4 frame rate"},
{"name": "--keep_frames", "action": "store_true", "default": False, "help": "Keep intermediate PNG frames after mp4 export"},
```

```python
# play.py
recording = prepare_recording(args.record, args.record_dir or default_dir, args.record_name)
...
if recording.enabled:
    env.gym.write_viewer_image_to_file(env.viewer, recording.frame_pattern % frame_idx)
...
video_path = finalize_recording(recording, args.record_fps, args.keep_frames)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recording.py -v`
Expected: PASS

### Task 3: Validate usage and failure paths

**Files:**
- Modify: `legged_gym/scripts/play.py`
- Test: `tests/test_recording.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest


def test_finalize_recording_raises_when_enabled_but_no_frames_exist(tmp_path):
    recording = prepare_recording(True, tmp_path, "demo")

    with pytest.raises(RuntimeError, match="No recorded frames"):
        finalize_recording(recording, fps=30, keep_frames=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recording.py::test_finalize_recording_raises_when_enabled_but_no_frames_exist -v`
Expected: FAIL until the empty-recording guard exists.

- [ ] **Step 3: Write minimal implementation**

```python
frame_files = sorted(recording.frames_dir.glob("frame_*.png"))
if not frame_files:
    raise RuntimeError("No recorded frames were written; cannot create mp4.")
```

- [ ] **Step 4: Run tests and targeted verification**

Run: `pytest tests/test_recording.py -v`
Expected: PASS

Run: `python -m compileall legged_gym/scripts/play.py legged_gym/utils/helpers.py legged_gym/utils/recording.py`
Expected: PASS
