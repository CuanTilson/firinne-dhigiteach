def fuse_forensic_scores(
    ml_prob: float,
    metadata_anomaly: float,
    c2pa_score: float,
    c2pa_ai_flag: bool,
    ela_score: float,
    noise_score: float,
    qtable_score: float,
    sd_watermark_score: float,  # NEW
    weights: dict | None = None,
    classification_bands: dict | None = None,
) -> dict:
    """
    Weighted fusion of:
      - ML model probability
      - EXIF/metadata anomaly score
      - C2PA soft confidence
      - ELA anomaly score
      - Noise residual anomaly
      - JPEG quantisation anomaly
      - Stable Diffusion invisible watermark presence
    """

    # Explicit AI from C2PA (strongest evidence available)
    if c2pa_ai_flag:
        final_score = max(ml_prob, c2pa_score, 0.9)
        return {
            "final_score": float(final_score),
            "classification": "ai_generated_c2pa_flagged",
            "override": True,
        }

    # --- Recommended weights ---
    # Total = 1.00
    weights = weights or {}
    w_ml = float(weights.get("ml", 0.50))
    w_meta = float(weights.get("metadata", 0.10))
    w_c2pa = float(weights.get("c2pa", 0.10))
    w_ela = float(weights.get("ela", 0.10))
    w_noise = float(weights.get("noise", 0.10))
    w_q = float(weights.get("jpeg", 0.05))
    w_sdwm = float(weights.get("sd_watermark", 0.02))  # Optional invisible watermark is a weak supporting cue

    final_score = (
        (ml_prob * w_ml)
        + (metadata_anomaly * w_meta)
        + (c2pa_score * w_c2pa)
        + (ela_score * w_ela)
        + (noise_score * w_noise)
        + (qtable_score * w_q)
        + (sd_watermark_score * w_sdwm)  # NEW
    )

    # --- Classification bands ---
    classification_bands = classification_bands or {}
    ai_likely_min = float(classification_bands.get("ai_likely_min", 0.7))
    real_likely_max = float(classification_bands.get("real_likely_max", 0.3))
    if final_score >= ai_likely_min:
        label = "likely_ai_generated"
    elif final_score <= real_likely_max:
        label = "likely_real"
    else:
        label = "uncertain"

    return {
        "final_score": float(final_score),
        "classification": label,
        "override": False,
    }
