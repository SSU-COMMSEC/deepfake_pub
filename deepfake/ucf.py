"""UCF-101 split-1 lists and video decoding."""

import cv2
import numpy as np
from pathlib import Path

from deepfake.paths import UCF_ROOT, SPLIT_DIR


def split_videos(split: str) -> list[str]:
    """Relative paths ('Class/v_Class_gXX_cYY.avi') of the official split-1 train or test list."""
    name = {"train": "trainlist01.txt", "test": "testlist01.txt"}[split]
    out = []
    for line in open(SPLIT_DIR / name):
        line = line.strip()
        if line:
            out.append(line.split()[0])
    return out


def group_of(rel: str) -> int:
    """UCF-101 group number (the same group usually means the same actor and scene)."""
    return int(Path(rel).stem.split("_")[-2][1:])


def read_frames(rel_or_path: str | Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    path = Path(rel_or_path)
    if not path.is_absolute():
        path = UCF_ROOT / path
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, float(fps)


def write_video(path: str | Path, frames_bgr: list[np.ndarray], fps: float, fourcc: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot encode {fourcc} into {path}")
    for f in frames_bgr:
        writer.write(np.ascontiguousarray(f))
    writer.release()
    return path


def write_previews(stem: str | Path, frames_bgr: list[np.ndarray], fps: float) -> list[Path]:
    """Same clip as VP9 .webm (VS Code / browser) and MPEG-4 .mp4 (desktop players)."""
    return [write_video(Path(f"{stem}{ext}"), frames_bgr, fps, cc) for cc, ext in (("VP90", ".webm"), ("mp4v", ".mp4"))]
