from pathlib import Path
import numpy as np
import cv2
import os
from PIL import Image, ImageChops, ExifTags
import uuid

# Optional import for SD invisible watermark detection
try:
    from imwatermark import WatermarkDecoder
except ImportError:
    WatermarkDecoder = None


# ============================================================
#  ORIENTATION FIX
# ============================================================


def apply_orientation(img: Image.Image) -> Image.Image:
    """Corrects image orientation based on EXIF."""
    try:
        exif = img._getexif()
        if not exif:
            return img

        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orientation_key is None:
            return img

        orientation = exif.get(orientation_key)

        if orientation == 3:
            return img.rotate(180, expand=True)
        elif orientation == 6:  # 90° CW
            return img.rotate(270, expand=True)
        elif orientation == 8:  # 90° CCW
            return img.rotate(90, expand=True)

        return img
    except Exception:
        return img


# ============================================================
#  1. NOISE ANALYSIS
# ============================================================


def _load_gray(image_path: Path):
    img = Image.open(image_path)
    img = apply_orientation(img)
    img = img.convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def _noise_residual(image: np.ndarray):
    """Simple high-pass filter (Laplacian) to extract sensor noise."""
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def _frequency_spectrum(image: np.ndarray):
    """Magnitude of FFT spectrum."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return np.abs(fshift)


def analyse_noise(image_path: Path) -> dict:
    img = _load_gray(image_path)

    # Noise residual
    res = _noise_residual(img)
    variance = float(np.var(res))

    # Frequency behaviour
    freq = _frequency_spectrum(img)
    freq_norm = freq / (np.max(freq) + 1e-8)
    flatness = float(np.mean(freq_norm))

    # Heuristic scoring
    score = 0.0

    if variance < 0.0005:
        score += 0.6
    elif variance < 0.001:
        score += 0.3

    if flatness > 0.25:
        score += 0.4
    elif flatness > 0.18:
        score += 0.2

    return {
        "residual_variance": variance,
        "spectral_flatness": flatness,
        "noise_anomaly_score": min(1.0, score),
    }


def generate_noise_heatmap(
    image_path: Path,
    save_path: Path | None = None,
    window_size: int = 16,
) -> dict:
    img = _load_gray(image_path)
    residual = _noise_residual(img)

    local_var = cv2.blur(residual ** 2, (window_size, window_size))
    local_var = np.sqrt(np.clip(local_var, 0, None))

    min_val = float(local_var.min())
    max_val = float(local_var.max())
    mean_val = float(local_var.mean())

    heatmap_path = None
    if save_path is not None:
        norm = (local_var - min_val) / (max_val - min_val + 1e-8)
        heat = (norm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        Image.fromarray(heat_rgb).save(save_path)
        heatmap_path = str(save_path)

    return {
        "local_variance_min": min_val,
        "local_variance_max": max_val,
        "local_variance_mean": mean_val,
        "noise_heatmap_path": heatmap_path,
    }


# ============================================================
#  2. JPEG QTABLE ANALYSIS
# ============================================================

STANDARD_LIBJPEG_TABLES = {
    "std_luma": np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    ),
    "std_chroma": np.array(
        [
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ]
    ),
}


def _extract_qtables(image_path: Path):
    try:
        img = Image.open(image_path)
        if not hasattr(img, "quantization"):
            return None
        return img.quantization
    except Exception:
        return None


def _score_qtables(qtables) -> float:
    if qtables is None:
        return 0.0

    tables = list(qtables.values())
    if len(tables) == 0:
        return 0.0

    luma = np.array(tables[0]).reshape(8, 8)

    d_std = np.mean(np.abs(luma - STANDARD_LIBJPEG_TABLES["std_luma"])) / 255.0
    variance = float(np.var(luma) / 5000)

    return float(min(1.0, (d_std * 0.7) + (variance * 0.3)))


def analyse_qtables(image_path: Path):
    q = _extract_qtables(image_path)
    anomaly = _score_qtables(q)
    return {
        "qtables_found": q is not None,
        "qtables": q,
        "qtables_anomaly_score": anomaly,
    }


def _estimate_quality_from_table(qtable) -> int | None:
    try:
        luma = np.array(qtable).reshape(8, 8)
    except Exception:
        return None

    std = STANDARD_LIBJPEG_TABLES["std_luma"]
    ratios = luma / (std + 1e-8)
    scale = float(np.median(ratios) * 100.0)

    if scale <= 0:
        return None

    if scale <= 100:
        quality = int(round(5000 / scale))
    else:
        quality = int(round((200 - scale) / 2))

    return int(min(100, max(1, quality)))


def estimate_jpeg_quality(image_path: Path) -> dict:
    qtables = _extract_qtables(image_path)
    if qtables is None:
        return {"quality_estimate": None}

    tables = list(qtables.values())
    if not tables:
        return {"quality_estimate": None}

    quality = _estimate_quality_from_table(tables[0])
    return {"quality_estimate": quality}


def jpeg_double_compression_heatmap(
    image_path: Path,
    save_path: Path | None = None,
    qualities: list[int] | None = None,
    max_size: int = 640,
) -> dict:
    qualities = qualities or [60, 70, 80, 90]
    jpeg_path = _ensure_jpeg(image_path)

    with Image.open(jpeg_path) as img:
        img = apply_orientation(img).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_size / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))

        original = np.array(img, dtype=np.float32)

    diffs = []
    tmp_paths = []
    for q in qualities:
        tmp_path = jpeg_path.with_suffix(f".q{q}.{uuid.uuid4().hex}.jpg")
        tmp_paths.append(tmp_path)
        Image.fromarray(original.astype(np.uint8)).save(tmp_path, "JPEG", quality=q)
        recompressed = np.array(Image.open(tmp_path).convert("RGB"), dtype=np.float32)
        diff = np.mean(np.abs(original - recompressed), axis=2)
        diffs.append(diff)

    diff_stack = np.stack(diffs, axis=2)
    best_idx = np.argmin(diff_stack, axis=2)
    best_quality = np.vectorize(lambda i: qualities[i])(best_idx)
    flat = best_quality.flatten()
    values, counts = np.unique(flat, return_counts=True)
    mode_quality = int(values[np.argmax(counts)]) if len(values) else qualities[0]
    inconsistency = float(np.mean(best_quality != mode_quality))

    heatmap_path = None
    if save_path is not None:
        diff_map = np.abs(best_quality - mode_quality).astype(np.float32)
        norm = diff_map / (np.max(diff_map) + 1e-8)
        heat = (norm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_MAGMA)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        Image.fromarray(heat_rgb).save(save_path)
        heatmap_path = str(save_path)

    for tmp in tmp_paths:
        try:
            os.remove(tmp)
        except OSError:
            pass

    return {
        "mode_quality": mode_quality,
        "inconsistency_score": inconsistency,
        "jpeg_quality_heatmap_path": heatmap_path,
    }


# ============================================================
#  3. ERROR LEVEL ANALYSIS (ELA) WITH ORIENTATION FIX
# ============================================================


def _ensure_jpeg(src_path: Path) -> Path:
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
    save_path: Path | None = None,
) -> dict:
    jpeg_path = _ensure_jpeg(image_path)

    # Load + correct orientation
    with Image.open(jpeg_path) as img:
        img = apply_orientation(img).convert("RGB")
        original = img

        recompressed_path = jpeg_path.with_suffix(".ela_recompressed.jpg")
        original.save(recompressed_path, "JPEG", quality=quality)

    # Difference
    with Image.open(jpeg_path) as img1, Image.open(recompressed_path) as img2:
        img1 = apply_orientation(img1).convert("RGB")
        img2 = apply_orientation(img2).convert("RGB")

        diff = ImageChops.difference(img1, img2)

        diff_np = np.array(diff).astype(np.float32)
        diff_np *= scale_factor
        diff_np = np.clip(diff_np, 0, 255).astype(np.uint8)
        diff_enhanced = Image.fromarray(diff_np)

        mean_error = float(diff_np.mean())
        max_error = float(diff_np.max())

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


# ============================================================
#  4. SD INVISIBLE WATERMARK
# ============================================================


def detect_sd_watermark(image_path: Path) -> dict:
    if WatermarkDecoder is None:
        return {
            "watermark_detected": False,
            "confidence": 0.0,
            "error": "invisible-watermark not installed",
        }

    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)

        decoder = WatermarkDecoder("bytes", 4)
        watermark = decoder.decode(arr, "dwtDct")

        if watermark is None:
            return {
                "watermark_detected": False,
                "confidence": 0.0,
                "error": None,
            }

        wm_str = watermark.decode(errors="ignore")

        hits = ("SD" in wm_str) + ("SDW" in wm_str) + ("sd" in wm_str.lower())
        confidence = min(1.0, hits / 3)

        return {
            "watermark_detected": confidence > 0.3,
            "confidence": confidence,
            "raw_watermark_string": wm_str,
            "error": None,
        }

    except Exception as e:
        return {
            "watermark_detected": False,
            "confidence": 0.0,
            "error": str(e),
        }
