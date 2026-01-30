from pathlib import Path
from datetime import datetime
import hashlib
import exifread
import re
from pymediainfo import MediaInfo

# ============================================================
#  SECTION 1 — METADATA EXTRACTION (IMAGE + VIDEO)
# ============================================================

def extract_image_metadata(file_path: Path) -> dict:
    """Extract EXIF metadata from an image file."""
    metadata = {}
    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=True)

        for tag, value in tags.items():
            metadata[tag] = str(value)

    except Exception as e:
        metadata["error"] = str(e)

    return metadata


def extract_video_metadata(file_path: Path) -> dict:
    """Extract metadata from a video using MediaInfo."""
    metadata = {}
    try:
        media_info = MediaInfo.parse(file_path)
        for track in media_info.tracks:
            metadata[track.track_type] = track.to_data()
    except Exception as e:
        metadata["error"] = str(e)

    return metadata


# ============================================================
#  SECTION 2 — METADATA ANOMALY ANALYSIS
# ============================================================

AI_PATTERNS = [
    r"\bstable diffusion\b",
    r"\bmidjourney\b",
    r"\bdall[- ]?e\b",
    r"\brunwayml\b",
    r"\bcomfyui\b",
    r"\bnovelai\b",
    r"\bai\b",
    r"\bgenerated\b",
]

SAFE_TAGS = [
    "Image Software",
    "Image Model",
    "Image Make",
    "EXIF UserComment",
    "EXIF ImageDescription",
    "EXIF Software",
    "EXIF ProcessingSoftware",
]

CAMERA_MAKE_WHITELIST = [
    "canon",
    "nikon",
    "sony",
    "fujifilm",
    "panasonic",
    "olympus",
    "pentax",
    "leica",
    "samsung",
    "apple",
    "google",
    "xiaomi",
    "huawei",
    "oppo",
    "vivo",
    "oneplus",
]


def _is_binary_string(s: str) -> bool:
    """Rough heuristic to detect non-text EXIF fields."""
    ascii_printables = sum(32 <= ord(c) <= 126 for c in s)
    return ascii_printables < (len(s) * 0.6)


def analyse_image_metadata(meta: dict) -> dict:
    """Basic heuristic scoring for EXIF metadata consistency."""
    findings = []
    score = 0.0

    # Missing EXIF
    if len(meta) == 0 or "EXIF DateTimeOriginal" not in meta:
        findings.append("Missing or minimal EXIF data")
        score += 0.3

    # Scan "safe" fields only
    for tag, value in meta.items():
        if tag not in SAFE_TAGS:
            continue

        value_str = str(value)
        if _is_binary_string(value_str):
            continue

        for pattern in AI_PATTERNS:
            if re.search(pattern, value_str.lower()):
                findings.append(f"AI-related software tag detected in: {tag}")
                score += 0.5

    # Missing camera model
    if "Image Model" not in meta:
        findings.append("No camera model in metadata")
        score += 0.2

    return {
        "anomaly_score": min(score, 1.0),
        "findings": findings,
    }


def check_camera_model_consistency(meta: dict) -> dict:
    warnings = []
    score = 0.0

    make_raw = str(meta.get("Image Make", "") or "").strip()
    model_raw = str(meta.get("Image Model", "") or "").strip()
    make = make_raw.lower()
    model = model_raw.lower()

    if not make_raw:
        warnings.append("Missing camera make in metadata")
        score += 0.2

    if not model_raw:
        warnings.append("Missing camera model in metadata")
        score += 0.2

    if make and not any(m in make for m in CAMERA_MAKE_WHITELIST):
        warnings.append("Unknown camera manufacturer in metadata")
        score += 0.2

    if model_raw and len(model_raw) < 2:
        warnings.append("Suspiciously short camera model")
        score += 0.1

    if make and model and any(m in make for m in CAMERA_MAKE_WHITELIST):
        if make not in model and make not in ("apple", "google"):
            warnings.append("Camera model does not include manufacturer name")
            score += 0.1

    return {
        "score": min(score, 1.0),
        "warnings": warnings,
        "make": make_raw or None,
        "model": model_raw or None,
    }


# ============================================================
#  SECTION 3 — EXIF FORENSICS (ADVANCED CHECKING)
# ============================================================

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
    "samsung", "apple", "google", "sony", "xiaomi",
    "oneplus", "huawei", "oppo", "vivo",
]


def _safe_int(value: str):
    try:
        return int(value)
    except Exception:
        return None


def exif_forensics(metadata: dict) -> dict:
    """Deep EXIF consistency checking."""
    warnings = []
    score = 0.0

    # Missing important fields
    missing = [k for k in EXPECTED_KEYS if k not in metadata]
    if missing:
        warnings.append(f"Missing important EXIF fields: {missing}")
        score += 0.15

    # Size mismatches
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

    # Camera make/model validity
    make = metadata.get("Image Make", "").lower()
    model = metadata.get("Image Model", "").lower()

    if not make or not model:
        warnings.append("Camera make/model missing")
        score += 0.20

    # Smartphone but missing MakerNote
    if any(mk in make for mk in SMARTPHONE_MAKERNOTE_PRESENT):
        if "EXIF MakerNote" not in metadata:
            warnings.append("Expected smartphone MakerNote is missing")
            score += 0.25

    # Timestamp validity
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

    # GPS consistency
    gps_lat = metadata.get("GPS GPSLatitude")
    gps_lon = metadata.get("GPS GPSLongitude")

    if (gps_lat and not gps_lon) or (gps_lon and not gps_lat):
        warnings.append("GPS metadata incomplete (lat or lon missing)")
        score += 0.10

    return {
        "warnings": warnings,
        "score": min(score, 1.0),
    }


# ============================================================
#  SECTION 4 — FILE-INTEGRITY / STRUCTURE FORENSICS
# ============================================================

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


def file_hashes(path: Path) -> dict:
    """Return MD5 + SHA256 checksums."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()

    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
            md5.update(chunk)

    return {
        "sha256": sha256.hexdigest(),
        "md5": md5.hexdigest(),
    }


def analyse_jpeg_structure(path: Path) -> dict:
    """Check JPEG structural markers and recompression signatures."""
    result = {
        "valid_jpeg": True,
        "missing_soi": False,
        "missing_eoi": False,
        "double_compressed": False,
        "app1_segments": 0,
        "warnings": [],
    }

    try:
        data = Path(path).read_bytes()
    except Exception as e:
        return {"valid_jpeg": False, "error": str(e)}

    # SOI/EOI markers
    if not data.startswith(JPEG_SOI):
        result["valid_jpeg"] = False
        result["missing_soi"] = True
        result["warnings"].append("Missing SOI marker.")

    if not data.endswith(JPEG_EOI):
        result["valid_jpeg"] = False
        result["missing_eoi"] = True
        result["warnings"].append("Missing EOI marker.")

    # Count EXIF/XMP segments
    app1_count = data.count(b"\xff\xe1")
    result["app1_segments"] = app1_count

    if app1_count == 0:
        result["warnings"].append("No APP1 (EXIF/XMP) segment found.")
    elif app1_count > 2:
        result["warnings"].append("Suspicious number of APP1 segments.")

    # Double compression indicator: multiple DQT tables
    if data.count(b"\xff\xdb") >= 3:
        result["double_compressed"] = True
        result["warnings"].append("Multiple DQT segments — likely double compressed.")

    return result


def analyse_file_integrity(path: Path) -> dict:
    return {
        "hashes": file_hashes(path),
        "jpeg_structure": analyse_jpeg_structure(path)
        if path.suffix.lower() in [".jpg", ".jpeg"]
        else {},
    }
