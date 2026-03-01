import json
from pathlib import Path

from backend.database.db import SessionLocal
from backend.database.models import AnalysisRecord, AudioAnalysisRecord, VideoAnalysisRecord


def _sample_ids(rows, limit: int = 10) -> list[int]:
    return [row.id for row in rows[:limit]]


def _image_checks(rows: list[AnalysisRecord]) -> dict:
    missing_detector = []
    missing_fusion_provenance = []
    missing_decision_path = []
    recoverable_from_settings = []
    for row in rows:
        ml_prediction = row.ml_prediction if isinstance(row.ml_prediction, dict) else {}
        detector = ml_prediction.get("detector") if isinstance(ml_prediction.get("detector"), dict) else {}
        forensic_score_json = row.forensic_score_json if isinstance(row.forensic_score_json, dict) else {}
        provenance = (
            forensic_score_json.get("provenance")
            if isinstance(forensic_score_json.get("provenance"), dict)
            else {}
        )
        decision_path = (
            forensic_score_json.get("decision_path")
            if isinstance(forensic_score_json.get("decision_path"), dict)
            else {}
        )
        applied_settings = row.applied_settings if isinstance(row.applied_settings, dict) else {}
        pipeline_settings = (
            applied_settings.get("pipeline")
            if isinstance(applied_settings.get("pipeline"), dict)
            else {}
        )
        if not detector:
            missing_detector.append(row)
            if pipeline_settings.get("image_detector"):
                recoverable_from_settings.append(row)
        if not provenance:
            missing_fusion_provenance.append(row)
        if not decision_path:
            missing_decision_path.append(row)
    return {
        "total": len(rows),
        "missing_detector": len(missing_detector),
        "missing_detector_sample_ids": _sample_ids(missing_detector),
        "missing_fusion_provenance": len(missing_fusion_provenance),
        "missing_fusion_provenance_sample_ids": _sample_ids(missing_fusion_provenance),
        "missing_decision_path": len(missing_decision_path),
        "missing_decision_path_sample_ids": _sample_ids(missing_decision_path),
        "recoverable_from_applied_settings": len(recoverable_from_settings),
        "recoverable_from_applied_settings_sample_ids": _sample_ids(recoverable_from_settings),
    }


def _video_checks(rows: list[VideoAnalysisRecord]) -> dict:
    missing_detector = []
    recoverable_from_settings = []
    for row in rows:
        video_metadata = row.video_metadata if isinstance(row.video_metadata, dict) else {}
        detector = (
            video_metadata.get("image_detector")
            if isinstance(video_metadata.get("image_detector"), dict)
            else {}
        )
        applied_settings = row.applied_settings if isinstance(row.applied_settings, dict) else {}
        pipeline_settings = (
            applied_settings.get("pipeline")
            if isinstance(applied_settings.get("pipeline"), dict)
            else {}
        )
        if not detector:
            missing_detector.append(row)
            if pipeline_settings.get("image_detector"):
                recoverable_from_settings.append(row)
    return {
        "total": len(rows),
        "missing_image_detector": len(missing_detector),
        "missing_image_detector_sample_ids": _sample_ids(missing_detector),
        "recoverable_from_applied_settings": len(recoverable_from_settings),
        "recoverable_from_applied_settings_sample_ids": _sample_ids(recoverable_from_settings),
    }


def _audio_checks(rows: list[AudioAnalysisRecord]) -> dict:
    missing_settings = []
    for row in rows:
        if not isinstance(row.applied_settings, dict):
            missing_settings.append(row)
    return {
        "total": len(rows),
        "missing_applied_settings": len(missing_settings),
        "missing_applied_settings_sample_ids": _sample_ids(missing_settings),
    }


def main() -> None:
    db = SessionLocal()
    try:
        image_rows = db.query(AnalysisRecord).order_by(AnalysisRecord.id.asc()).all()
        video_rows = db.query(VideoAnalysisRecord).order_by(VideoAnalysisRecord.id.asc()).all()
        audio_rows = db.query(AudioAnalysisRecord).order_by(AudioAnalysisRecord.id.asc()).all()
    finally:
        db.close()

    report = {
        "images": _image_checks(image_rows),
        "videos": _video_checks(video_rows),
        "audio": _audio_checks(audio_rows),
    }

    output_dir = Path("artifacts") / "week5_provenance_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "provenance_audit_summary.json"
    md_path = output_dir / "provenance_audit_summary.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Week 5 Provenance Audit",
        "",
        "## Images",
        f"- Total records: {report['images']['total']}",
        f"- Missing detector metadata: {report['images']['missing_detector']}",
        f"- Missing fusion provenance: {report['images']['missing_fusion_provenance']}",
        f"- Missing decision path: {report['images']['missing_decision_path']}",
        f"- Recoverable from applied settings: {report['images']['recoverable_from_applied_settings']}",
        "",
        "## Videos",
        f"- Total records: {report['videos']['total']}",
        f"- Missing image detector metadata: {report['videos']['missing_image_detector']}",
        f"- Recoverable from applied settings: {report['videos']['recoverable_from_applied_settings']}",
        "",
        "## Audio",
        f"- Total records: {report['audio']['total']}",
        f"- Missing applied settings snapshot: {report['audio']['missing_applied_settings']}",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
