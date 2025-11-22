def fuse_forensic_scores(
    ml_prob: float,
    metadata_anomaly: float,
    c2pa_score: float,
    c2pa_ai_flag: bool,
) -> dict:
    """
    ml_prob: model probability the image is AI-generated (0-1)
    metadata_anomaly: anomaly score from metadata analyser (0-1)
    c2pa_score: 0-1 AI likelihood from C2PA (soft signals)
    c2pa_ai_flag: True if C2PA explicitly indicates AI was used
    """

    # Explicit C2PA AI signal
    if c2pa_ai_flag:
        final_score = max(ml_prob, c2pa_score, 0.9)
        return {
            "final_score": float(final_score),
            "classification": "ai_generated_c2pa_flagged",
            "override": True,
        }

    # Soft blend
    final_score = (ml_prob * 0.7) + (metadata_anomaly * 0.1) + (c2pa_score * 0.2)

    if final_score > 0.7:
        label = "likely_ai_generated"
    elif final_score < 0.3:
        label = "likely_real"
    else:
        label = "uncertain"

    return {
        "final_score": float(final_score),
        "classification": label,
        "override": False,
    }
