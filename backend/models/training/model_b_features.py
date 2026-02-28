from __future__ import annotations

from typing import Any


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_flag(value: Any) -> int:
    return 1 if bool(value) else 0


def extract_model_b_features_from_record(record) -> dict[str, float | int | str | None]:
    metadata_anomalies = _as_dict(getattr(record, "metadata_anomalies", None))
    file_integrity = _as_dict(getattr(record, "file_integrity", None))
    ai_watermark = _as_dict(getattr(record, "ai_watermark", None))
    c2pa = _as_dict(getattr(record, "c2pa", None))
    jpeg_qtables = _as_dict(getattr(record, "jpeg_qtables", None))
    noise_residual = _as_dict(getattr(record, "noise_residual", None))
    ela_analysis = _as_dict(getattr(record, "ela_analysis", None))

    ai_assertions = c2pa.get("ai_assertions_found")
    ai_assertion_count = len(ai_assertions) if isinstance(ai_assertions, list) else 0

    c2pa_ai_flag = (
        bool(c2pa.get("has_c2pa"))
        and (
            _as_float(c2pa.get("overall_c2pa_score")) >= 0.95
            or ai_assertion_count > 0
            or "synthetic" in [str(x).lower() for x in c2pa.get("digital_source_types", [])]
            or "photo assist" in [str(x).lower() for x in c2pa.get("software_agents", [])]
        )
    )

    return {
        "record_id": getattr(record, "id", None),
        "filename": getattr(record, "filename", None),
        "created_at": getattr(record, "created_at", None).isoformat()
        if getattr(record, "created_at", None) is not None
        else None,
        "classification": getattr(record, "classification", None),
        "rule_forensic_score": _as_float(getattr(record, "forensic_score", 0.0)),
        "ml_probability": _as_float(getattr(record, "ml_probability", 0.0)),
        "metadata_anomaly_score": _as_float(metadata_anomalies.get("anomaly_score")),
        "metadata_findings_count": len(metadata_anomalies.get("findings", []) or []),
        "hashes_match": _as_flag(file_integrity.get("hashes_match")),
        "jpeg_structure_valid": _as_flag(
            _as_dict(file_integrity.get("jpeg_structure")).get("valid_jpeg")
        ),
        "c2pa_present": _as_flag(c2pa.get("has_c2pa")),
        "c2pa_signature_valid": _as_flag(c2pa.get("signature_valid")),
        "c2pa_score": _as_float(c2pa.get("overall_c2pa_score")),
        "c2pa_ai_assertion_count": ai_assertion_count,
        "c2pa_ai_flag": _as_flag(c2pa_ai_flag),
        "jpeg_qtable_anomaly_score": _as_float(jpeg_qtables.get("qtables_anomaly_score")),
        "jpeg_quality_estimate": _as_float(jpeg_qtables.get("quality_estimate")),
        "jpeg_double_compression": _as_flag(jpeg_qtables.get("double_compression")),
        "jpeg_inconsistency_score": _as_float(jpeg_qtables.get("inconsistency_score")),
        "jpeg_combined_anomaly_score": _as_float(
            jpeg_qtables.get("combined_anomaly_score")
        ),
        "noise_variance": _as_float(noise_residual.get("residual_variance")),
        "noise_spectral_flatness": _as_float(noise_residual.get("spectral_flatness")),
        "noise_anomaly_score": _as_float(noise_residual.get("noise_anomaly_score")),
        "noise_inconsistency_score": _as_float(noise_residual.get("inconsistency_score")),
        "ela_mean_error": _as_float(ela_analysis.get("mean_error")),
        "ela_max_error": _as_float(ela_analysis.get("max_error")),
        "ela_anomaly_score": _as_float(ela_analysis.get("ela_anomaly_score")),
        "watermark_detected": _as_flag(
            ai_watermark.get("watermark_detected")
            or ai_watermark.get("stable_diffusion_detected")
        ),
        "watermark_confidence": _as_float(ai_watermark.get("confidence")),
    }
