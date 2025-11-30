import json
import subprocess
import sys
from pathlib import Path

# Path to vendor repo
BASE_DIR = Path(__file__).resolve().parents[2]
CNNDET_DIR = BASE_DIR / "vendor" / "CNNDetection"

DEMO_PY = CNNDET_DIR / "demo.py"
DEFAULT_WEIGHTS = CNNDET_DIR / "weights" / "blur_jpg_prob0.5.pth"


def run_cnndetection(image_path: Path, weights_path: Path = DEFAULT_WEIGHTS) -> dict:
    """
    Calls CNNDetection's demo.py via subprocess and extracts the score.
    Returns a dict: {label, score, model, raw_output}
    """

    if not DEMO_PY.exists():
        raise RuntimeError(f"demo.py not found at {DEMO_PY}")

    # convert paths to absolute (fixes your error)
    image_path = image_path.resolve()
    weights_path = weights_path.resolve()

    if not weights_path.exists():
        raise RuntimeError(f"Model weights not found at {weights_path}")

    cmd = [
        sys.executable,
        str(DEMO_PY),
        "-f",
        str(image_path),
        "-m",
        str(weights_path),
        # NOTE: no --use_cpu here (GPU enabled)
    ]

    proc = subprocess.run(
        cmd, cwd=str(CNNDET_DIR), capture_output=True, text=True  # run from vendor dir
    )

    if proc.returncode != 0:
        raise RuntimeError(f"CNNDetection failed: {proc.stderr}")

    raw_output = proc.stdout.strip()

    # Extract score (float) from output
    # Parse CNNDetection output
    # Expected line example: "probability of being synthetic: 0.73%"
    score = None
    for line in raw_output.splitlines():
        line = line.strip().lower()
        if "probability of being synthetic" in line:
            # Extract the numeric part before the '%'
            try:
                percent_value = line.split(":")[1].strip().replace("%", "")
                score = float(percent_value) / 100.0  # convert to 0.0–1.0
            except Exception:
                pass

    if score is None:
        raise RuntimeError(
            f"Could not parse score from CNNDetection output: {raw_output}"
        )

    # > 0.5 means more likely synthetic
    label = "ai-generated" if score > 0.5 else "real"


    return {
        "model": "CNNDetection blur_jpg_prob0.5",
        "score": score,
        "label": label,
        "raw_output": raw_output,
    }
