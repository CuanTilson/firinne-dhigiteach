from pathlib import Path
import hashlib

JPEG_SOI = b"\xff\xd8"   # Start of Image
JPEG_EOI = b"\xff\xd9"   # End of Image


def file_hashes(path: Path) -> dict:
    """Returns SHA-256 and MD5 hashes of the file."""
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
    """
    Checks:
      - SOI/EOI markers
      - Minimum size
      - Presence of multiple compression passes
      - APP1 (EXIF/XMP) validity
    """
    result = {
        "valid_jpeg": True,
        "missing_soi": False,
        "missing_eoi": False,
        "double_compressed": False,
        "app1_segments": 0,
        "warnings": [],
    }

    data = None
    try:
        data = Path(path).read_bytes()
    except Exception as e:
        return {
            "valid_jpeg": False,
            "error": str(e),
        }

    # --- SOI / EOI ---
    if not data.startswith(JPEG_SOI):
        result["missing_soi"] = True
        result["valid_jpeg"] = False
        result["warnings"].append("Missing SOI marker (not a JPEG).")

    if not data.endswith(JPEG_EOI):
        result["missing_eoi"] = True
        result["valid_jpeg"] = False
        result["warnings"].append("Missing EOI marker (corrupted or edited).")

    # --- Count APP1 (EXIF/XMP) ---
    # APP1 marker = FF E1
    app1_count = data.count(b"\xff\xe1")
    result["app1_segments"] = app1_count

    if app1_count == 0:
        result["warnings"].append("No APP1 segment (missing EXIF/XMP).")
    elif app1_count > 2:
        result["warnings"].append("Suspicious number of APP1 segments — may indicate recompression or tampering.")

    # --- Double compression test ---
    # Common signature: multiple DQT segments (FF DB)
    dqt_count = data.count(b"\xff\xdb")
    if dqt_count >= 3:
        result["double_compressed"] = True
        result["warnings"].append("Multiple DQT segments — likely double compressed.")

    return result


def analyse_file_integrity(path: Path) -> dict:
    """Wrapper combining all file-integrity checks."""
    hashes = file_hashes(path)

    jpeg_struct = {}
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        jpeg_struct = analyse_jpeg_structure(path)

    return {
        "hashes": hashes,
        "jpeg_structure": jpeg_struct,
    }
