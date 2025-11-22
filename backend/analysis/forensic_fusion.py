def fuse_forensic_scores(ml_prob: float, metadata_anomaly: float) -> dict:
    """
    ml_prob: model probability the image is AI-generated (0–1)
    metadata_anomaly: anomaly score from metadata analyser (0–1)
    """

    final_score = (ml_prob * 0.7) + (metadata_anomaly * 0.3)

    if final_score > 0.7:
        label = "likely_ai_generated"
    elif final_score < 0.3:
        label = "likely_real"
    else:
        label = "uncertain"

    return {"final_score": float(final_score), "classification": label}
