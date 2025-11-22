# backend/analysis/ela.py
from PIL import Image, ImageChops, ImageEnhance
from pathlib import Path
import numpy as np


def compute_ela_score(image_path: Path, quality: int = 95):
    """
    Performs Error Level Analysis and returns:
      - score (0-1)
      - ela_image (np.ndarray)
    """
    try:
        original = Image.open(image_path).convert("RGB")

        # Save a recompressed version
        temp_path = image_path.with_suffix(".ela_tmp.jpg")
        original.save(temp_path, "JPEG", quality=quality)

        recompressed = Image.open(temp_path)

        # Difference between original and recompressed
        diff = ImageChops.difference(original, recompressed)

        # Increase contrast of difference
        diff = ImageEnhance.Brightness(diff).enhance(10)

        # Convert to numpy for scoring
        ela_array = np.array(diff).astype("float32")

        # Score: higher variation = more editing / AI generation likelihood
        # Normalised 0–1 based on pixel intensity spread
        ela_score = float(np.mean(ela_array) / 255.0)

        # Clean up tiny ELA temp file
        try:
            temp_path.unlink()
        except:
            pass

        return {
            "ela_score": ela_score,
            "ela_map": ela_array,  # used later for heatmap overlay
        }

    except Exception as e:
        return {
            "ela_score": None,
            "ela_map": None,
            "error": str(e),
        }
