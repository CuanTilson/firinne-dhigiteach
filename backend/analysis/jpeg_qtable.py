from pathlib import Path
from PIL import Image
import numpy as np

# Common quantisation tables used by cameras and by AI generators
STANDARD_LIBJPEG_TABLES = {
    "std_luma": np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99],
    ]),
    "std_chroma": np.array([
        [17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
    ]),
}

def extract_qtables(image_path: Path):
    """Return quantisation tables if image is JPEG."""
    try:
        img = Image.open(image_path)
        if not hasattr(img, "quantization"):
            return None
        return img.quantization
    except Exception:
        return None


def score_qtables(qtables) -> float:
    """
    Compute a 0-1 anomaly score:
      0 = camera-like tables
      1 = AI-like or generic tables
    """

    if qtables is None:
        return 0.0  # cannot score

    tables = list(qtables.values())
    if len(tables) == 0:
        return 0.0

    # Take only first two (luma + chroma)
    luma = np.array(tables[0]).reshape(8, 8)

    # 1) Compare with standard libjpeg tables
    d_std = np.mean(np.abs(luma - STANDARD_LIBJPEG_TABLES["std_luma"])) / 255.0

    # 2) Variance check — camera tables have certain distribution patterns
    variance = float(np.var(luma) / 5000)

    # Final score (clamped 0–1)
    score = min(1.0, (d_std * 0.7) + (variance * 0.3))

    return float(score)


def analyse_qtables(image_path: Path):
    q = extract_qtables(image_path)
    anomaly = score_qtables(q)

    return {
        "qtables_found": q is not None,
        "qtables": q,
        "qtables_anomaly_score": anomaly,
    }
