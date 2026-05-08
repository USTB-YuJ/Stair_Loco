from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import List, Optional, Union


def _find_ffmpeg() -> str:
    """Return ffmpeg path from the current Python env, falling back to PATH."""
    env_bin = Path(sys.executable).resolve().parent / "ffmpeg"
    if env_bin.is_file():
        return str(env_bin)
    found = shutil.which("ffmpeg")
    return found if found else "ffmpeg"


@dataclass(frozen=True)
class RecordingPaths:
    enabled: bool
    frames_dir: Optional[Path]
    frame_pattern: Optional[str]
    video_path: Optional[Path]


def prepare_recording(enabled: bool, record_dir: Union[str, Path], record_name: str) -> RecordingPaths:
    if not enabled:
        return RecordingPaths(
            enabled=False,
            frames_dir=None,
            frame_pattern=None,
            video_path=None,
        )

    base_dir = Path(record_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    record_stem = Path(record_name).name
    if record_stem.lower().endswith(".mp4"):
        record_stem = Path(record_stem).stem
    if not record_stem:
        record_stem = "play"

    frames_dir = (base_dir / f"{record_stem}_frames").resolve()
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    video_path = (base_dir / f"{record_stem}.mp4").resolve()
    if video_path.exists():
        video_path.unlink()

    return RecordingPaths(
        enabled=True,
        frames_dir=frames_dir,
        frame_pattern=str((frames_dir / "frame_%06d.png").resolve()),
        video_path=video_path,
    )


def finalize_recording(recording: RecordingPaths, fps: int, keep_frames: bool) -> Optional[Path]:
    if not recording.enabled:
        return None

    assert recording.frames_dir is not None
    assert recording.frame_pattern is not None
    assert recording.video_path is not None

    frame_files = _get_recorded_frames(recording)
    if not frame_files:
        raise RuntimeError("No recorded frames were written; cannot create mp4.")

    command = [
        _find_ffmpeg(),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        recording.frame_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(recording.video_path),
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        try:
            _encode_video_with_imageio(frame_files, recording.video_path, fps)
        except Exception:
            try:
                _encode_video_with_cv2(frame_files, recording.video_path, fps)
            except Exception:
                raise RuntimeError(f"ffmpeg failed to create mp4 at {recording.video_path}.") from exc
    except FileNotFoundError as exc:
        try:
            _encode_video_with_imageio(frame_files, recording.video_path, fps)
        except Exception:
            try:
                _encode_video_with_cv2(frame_files, recording.video_path, fps)
            except Exception:
                raise RuntimeError("ffmpeg is not installed or not on PATH; cannot create mp4.") from exc

    if not keep_frames:
        shutil.rmtree(recording.frames_dir)

    return recording.video_path


def finalize_recording_if_available(recording: RecordingPaths, fps: int, keep_frames: bool) -> Optional[Path]:
    if not recording.enabled:
        return None

    frame_files = _get_recorded_frames(recording)
    if not frame_files:
        return None

    return finalize_recording(recording, fps=fps, keep_frames=keep_frames)


def _encode_video_with_imageio(frame_files: List[Path], video_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is unavailable for mp4 fallback encoding.") from exc

    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame_path in frame_files:
            writer.append_data(imageio.imread(frame_path))


def _encode_video_with_cv2(frame_files: List[Path], video_path: Path, fps: int) -> None:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("cv2 is unavailable for mp4 fallback encoding.") from exc

    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        raise RuntimeError(f"Unable to read frame {frame_files[0]} for cv2 encoding.")

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open writer for {video_path}.")

    try:
        writer.write(first_frame)
        for frame_path in frame_files[1:]:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Unable to read frame {frame_path} for cv2 encoding.")
            writer.write(frame)
    finally:
        writer.release()


def _get_recorded_frames(recording: RecordingPaths) -> List[Path]:
    assert recording.frames_dir is not None
    return sorted(recording.frames_dir.glob("frame_*.png"))
