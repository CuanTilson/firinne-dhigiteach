from datetime import datetime
from typing import Dict, List

EXPECTED_KEYS = [
    "Image ImageWidth",
    "Image ImageLength",
    "EXIF ImageWidth",
    "EXIF ImageLength",
    "Image Make",
    "Image Model",
    "EXIF DateTimeOriginal",
    "EXIF FocalLength",
    "EXIF FocalLengthIn35mmFilm",
    "EXIF ISOSpeedRatings",
]

SMARTPHONE_MAKERNOTE_PRESENT = [
    "samsung",
    "apple",
    "google",
    "sony",
    "xiaomi",
    "oneplus",
    "huawei",
    "oppo",
    "vivo",
]


def _safe_int(value: str):
    try:
        return int(value)
    except Exception:
        return None


def exif_forensics(metadata: Dict[str, str]) -> Dict:
    warnings: List[str] = []
    score = 0.0

    # ---------------------------------------------------------
    # 1. Missing important EXIF fields
    # ---------------------------------------------------------
    missing = [k for k in EXPECTED_KEYS if k not in metadata]
    if missing:
        warnings.append(f"Missing important EXIF fields: {missing}")
        score += 0.15

    # ---------------------------------------------------------
    # 2. Pixel dimension inconsistencies (common in AI fillers)
    # ---------------------------------------------------------
    img_w = _safe_int(metadata.get("Image ImageWidth", ""))
    exif_w = _safe_int(metadata.get("EXIF ImageWidth", ""))
    img_h = _safe_int(metadata.get("Image ImageLength", ""))
    exif_h = _safe_int(metadata.get("EXIF ImageLength", ""))

    if img_w and exif_w and abs(img_w - exif_w) > 5:
        warnings.append("ImageWidth mismatch between Image and EXIF fields")
        score += 0.15

    if img_h and exif_h and abs(img_h - exif_h) > 5:
        warnings.append("ImageLength mismatch between Image and EXIF fields")
        score += 0.15

    # ---------------------------------------------------------
    # 3. Check Make/Model / Missing MakerNote (AI images rarely include it)
    # ---------------------------------------------------------
    make = metadata.get("Image Make", "").lower()
    model = metadata.get("Image Model", "").lower()

    if not make or not model:
        warnings.append("Camera make/model missing")
        score += 0.20

    # If it's a real phone make, it should have a maker note
    if any(mk in make for mk in SMARTPHONE_MAKERNOTE_PRESENT):
        if "EXIF MakerNote" not in metadata:
            warnings.append("Expected smartphone MakerNote is missing")
            score += 0.25

    # ---------------------------------------------------------
    # 4. Date/time validity (AI images often have nonsense timestamps)
    # ---------------------------------------------------------
    dt_raw = metadata.get("EXIF DateTimeOriginal")
    if dt_raw:
        try:
            dt = datetime.strptime(dt_raw, "%Y:%m:%d %H:%M:%S")
            if dt.year < 2000 or dt.year > 2035:
                warnings.append(f"Suspicious timestamp: {dt_raw}")
                score += 0.2
        except Exception:
            warnings.append(f"Invalid EXIF timestamp format: {dt_raw}")
            score += 0.20
    else:
        warnings.append("Missing EXIF DateTimeOriginal")
        score += 0.15

    # ---------------------------------------------------------
    # 5. GPS consistency
    # ---------------------------------------------------------
    gps_lat = metadata.get("GPS GPSLatitude")
    gps_lon = metadata.get("GPS GPSLongitude")

    if (gps_lat and not gps_lon) or (gps_lon and not gps_lat):
        warnings.append("GPS metadata incomplete (lat or lon missing)")
        score += 0.10

    # ---------------------------------------------------------
    # Final Score
    # ---------------------------------------------------------
    score = min(score, 1.0)

    return {
        "warnings": warnings,
        "score": score,
    }
