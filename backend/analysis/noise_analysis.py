import numpy as np
import cv2
from pathlib import Path
from PIL import Image


def load_gray(image_path: Path):
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def noise_residual(image: np.ndarray):
    """
    Simple high-pass filter (Laplacian) to extract noise residual.
    """
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    residual = cv2.filter2D(image, -1, kernel)
    return residual


def frequency_spectrum(image: np.ndarray):
    """
    Compute the magnitude of the FFT frequency spectrum.
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    return magnitude


def analyse_noise(image_path: Path) -> dict:
    """
    Returns:
      - residual_variance: high variance = camera-like, low variance = AI-like
      - freq_flatness: flatter frequency = AI-like
      - noise_anomaly_score: 0-1
    """
    img = load_gray(image_path)

    # --- Noise residual ---
    res = noise_residual(img)
    variance = float(np.var(res))

    # --- Frequency spectrum ---
    freq = frequency_spectrum(img)
    freq_norm = freq / (np.max(freq) + 1e-8)

    # Spectral flatness (entropy-like)
    # AI images tend to have flatter / smoother frequency distributions
    flatness = float(np.mean(freq_norm))

    # --- Heuristic scoring ---
    # Cameras → higher residual variance, lower flatness
    # AI → lower residual variance, high flatness

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
