from __future__ import annotations

from pathlib import Path
import uuid
from typing import List, Dict

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BACKEND_DIR / "storage"
VIDEO_FRAMES_DIR = STORAGE_DIR / "video_frames"
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


def _downsample_indices(indices: List[int], target: int) -> List[int]:
    if len(indices) <= target:
        return indices
    if target <= 0:
        return []
    pick = np.linspace(0, len(indices) - 1, num=target, dtype=int)
    return [indices[i] for i in pick]


def detect_scene_cuts(
    video_path: Path,
    sample_stride: int = 10,
    threshold: float = 0.6,
) -> List[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []

    prev_hist = None
    cuts = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diff = 1.0 - float(corr)
            if diff > threshold:
                cuts.append(frame_idx)

        prev_hist = hist
        frame_idx += 1

    cap.release()
    return cuts


def sample_video_frames(
    video_path: Path,
    max_frames: int = 16,
    output_dir: Path | None = None,
    scene_aware: bool = True,
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
    indices_set = set(indices.tolist())
    indices_set.update({0, total_frames - 1})

    priority = {0, total_frames - 1}
    if scene_aware:
        cuts = detect_scene_cuts(video_path)
        for cut in cuts:
            for offset in (-1, 0, 1):
                idx = max(0, min(total_frames - 1, cut + offset))
                indices_set.add(idx)
                priority.add(idx)

    indices = sorted(indices_set)

    if len(indices) > max_frames:
        prioritized = [i for i in indices if i in priority]
        if len(prioritized) > max_frames:
            prioritized = _downsample_indices(prioritized, max_frames)
            indices = prioritized
        else:
            remaining = [i for i in indices if i not in priority]
            slots = max_frames - len(prioritized)
            indices = prioritized + _downsample_indices(remaining, slots)

    indices = sorted(set(indices))

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
