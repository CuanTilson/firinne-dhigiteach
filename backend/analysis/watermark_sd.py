from pathlib import Path
import numpy as np
from PIL import Image

try:
    from imwatermark import WatermarkDecoder
except ImportError:
    WatermarkDecoder = None


def detect_sd_watermark(image_path: Path) -> dict:
    """
    Detects Stable Diffusion invisible watermark (DWT-DCT).

    Returns:
      {
        "watermark_detected": bool,
        "confidence": float,
        "error": str | None
      }
    """
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

        # Stable Diffusion watermark usually contains: "SDW" or "SDV1"
        hits = 0
        if "SD" in wm_str:
            hits += 1
        if "SDW" in wm_str:
            hits += 1
        if "sd" in wm_str.lower():
            hits += 1

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
