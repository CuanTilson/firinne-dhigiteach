from __future__ import annotations

from pathlib import Path
import uuid
from typing import List, Dict

import cv2
import numpy as np

VIDEO_FRAMES_DIR = Path("backend/storage/video_frames")
VIDEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def get_video_duration_seconds(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0.0

    total_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    if fps <= 0.0:
        return 0.0
    return total_frames / fps


def sample_video_frames(
    video_path: Path,
    max_frames: int = 16,
    output_dir: Path | None = None,
) -> List[Dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video for frame sampling.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Unable to determine video frame count.")

    output_dir = output_dir or (VIDEO_FRAMES_DIR / uuid.uuid4().hex)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = np.linspace(0, max(total_frames - 1, 0), num=max_frames, dtype=int)
    indices = sorted(set(indices.tolist()))

    frames: List[Dict] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame_path = output_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        timestamp = float(idx / fps) if fps else 0.0
        frames.append(
            {
                "frame_index": int(idx),
                "timestamp_sec": timestamp,
                "frame_path": frame_path,
            }
        )

    cap.release()
    return frames
