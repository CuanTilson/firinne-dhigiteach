import re

AI_PATTERNS = [
    r"\bstable diffusion\b",
    r"\bmidjourney\b",
    r"\bdall[- ]?e\b",
    r"\brunwayml\b",
    r"\bcomfyui\b",
    r"\bnovelai\b",
    r"\bai\b",  # whole word AI only
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


def is_binary_string(s: str) -> bool:
    # if string is mostly non-printable characters, treat as binary
    ascii_printables = sum(32 <= ord(c) <= 126 for c in s)
    return ascii_printables < (len(s) * 0.6)


def analyse_image_metadata(meta: dict) -> dict:
    findings = []
    score = 0.0

    # 1. Missing EXIF
    if len(meta) == 0 or "EXIF DateTimeOriginal" not in meta:
        findings.append("Missing or minimal EXIF data")
        score += 0.3

    # 2. Scan only safe tags, and only text-like values
    for tag, value in meta.items():
        if tag not in SAFE_TAGS:
            continue

        value_str = str(value)
        if is_binary_string(value_str):
            continue

        for pattern in AI_PATTERNS:
            if re.search(pattern, value_str.lower()):
                findings.append(f"AI-related software tag detected in: {tag}")
                score += 0.5

    # 3. No camera model
    if "Image Model" not in meta:
        findings.append("No camera model in metadata")
        score += 0.2

    return {"anomaly_score": min(score, 1.0), "findings": findings}
