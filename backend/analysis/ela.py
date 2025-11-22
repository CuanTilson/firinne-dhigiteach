from pathlib import Path
from typing import Optional
from PIL import Image, ImageChops
import numpy as np
import os


def _ensure_jpeg(src_path: Path) -> Path:
    """
    ELA works best on JPEG. If the file isn't JPEG, convert to a temp JPEG.
    Returns path to a JPEG file to use for ELA.
    """
    if src_path.suffix.lower() in [".jpg", ".jpeg"]:
        return src_path

    tmp_path = src_path.with_suffix(".ela_tmp.jpg")
    with Image.open(src_path).convert("RGB") as im:
        im.save(tmp_path, "JPEG", quality=95)
    return tmp_path


def perform_ela(
    image_path: Path,
    quality: int = 90,
    scale_factor: int = 20,
    save_path: Optional[Path] = None,
) -> dict:
    """
    Perform Error Level Analysis on an image.

    Returns:
      {
        "mean_error": float,
        "max_error": float,
        "ela_anomaly_score": float (0-1),
        "ela_image_path": str | None
      }
    """
    jpeg_path = _ensure_jpeg(image_path)

    # 1) Load original JPEG
    with Image.open(jpeg_path).convert("RGB") as original:
        # 2) Recompress
        recompressed_path = jpeg_path.with_suffix(".ela_recompressed.jpg")
        original.save(recompressed_path, "JPEG", quality=quality)

    with Image.open(jpeg_path).convert("RGB") as original, Image.open(
        recompressed_path
    ).convert("RGB") as recompressed:

        # 3) Pixelwise difference
        diff = ImageChops.difference(original, recompressed)

        # 4) Amplify differences
        diff_np = np.array(diff).astype(np.float32)
        diff_np *= scale_factor
        diff_np = np.clip(diff_np, 0, 255).astype(np.uint8)
        diff_enhanced = Image.fromarray(diff_np)

        # 5) Statistics
        mean_error = float(diff_np.mean())
        max_error = float(diff_np.max())

        # Simple heuristic scoring:
        #   very uniform / low error => possible generated / over-synthetic
        #   very high localised error => possible splicing / heavy edits
        score = 0.0

        if mean_error < 3.0:
            score += 0.4
        elif mean_error < 6.0:
            score += 0.2

        if max_error > 80.0:
            score += 0.4
        elif max_error > 50.0:
            score += 0.2

        ela_image_path = None
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            diff_enhanced.save(save_path, "PNG")
            ela_image_path = str(save_path)

    # clean up recompressed if you want
    try:
        os.remove(recompressed_path)
    except OSError:
        pass

    return {
        "mean_error": mean_error,
        "max_error": max_error,
        "ela_anomaly_score": float(min(score, 1.0)),
        "ela_image_path": ela_image_path,
    }
