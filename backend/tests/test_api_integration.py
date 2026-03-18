import io
import os
import unittest
from pathlib import Path
from unittest.mock import patch
import shutil

os.environ.setdefault("FD_ADMIN_KEY", "test-admin-key")

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.db import Base
from backend import main


def _write_image(path: Path, size: tuple[int, int] = (64, 64), color=(120, 80, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


class ApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_path = Path("backend/tests/.tmp_api_integration")
        if cls.temp_path.exists():
            shutil.rmtree(cls.temp_path, ignore_errors=True)
        cls.temp_path.mkdir(parents=True, exist_ok=True)

        cls.engine = create_engine(
            f"sqlite:///{(cls.temp_path / 'test_forensics.db').resolve().as_posix()}",
            connect_args={"check_same_thread": False},
        )
        cls.SessionTesting = sessionmaker(
            autocommit=False, autoflush=False, bind=cls.engine
        )
        Base.metadata.create_all(bind=cls.engine)

        cls.upload_dir = cls.temp_path / "uploaded"
        cls.thumb_dir = cls.temp_path / "thumbnails"
        cls.waveform_dir = cls.temp_path / "audio_plots"
        for directory in (cls.upload_dir, cls.thumb_dir, cls.waveform_dir):
            directory.mkdir(parents=True, exist_ok=True)

        def override_get_db():
            db = cls.SessionTesting()
            try:
                yield db
            finally:
                db.close()

        main.app.dependency_overrides[main.get_db] = override_get_db
        main.SessionLocal = cls.SessionTesting
        main.THUMB_DIR = cls.thumb_dir
        main.AUDIO_PLOTS_DIR = cls.waveform_dir
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls) -> None:
        main.app.dependency_overrides.clear()
        cls.engine.dispose()
        shutil.rmtree(cls.temp_path, ignore_errors=True)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    def _stub_save_image(self, file, max_bytes=None):
        target = self.upload_dir / "uploaded_image.jpg"
        _write_image(target)
        return target

    def _stub_save_audio(self, file, max_bytes=None):
        target = self.upload_dir / "uploaded_audio.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"RIFFtestWAVEfmt ")
        return target

    def _stub_save_video(self, file, max_bytes=None):
        target = self.upload_dir / "uploaded_video.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-video")
        return target

    def _create_audio_record(self) -> int:
        waveform = self.waveform_dir / "waveform.png"
        _write_image(waveform, size=(120, 40))

        analysis_payload = {
            "metadata": {"duration_seconds": 1.25, "sample_rate": 16000},
            "features": {
                "waveform_path": str(waveform),
                "analysis_mode": "waveform",
                "peak_level": 0.8,
            },
            "findings": ["Flat spectral profile"],
            "forensic_score": 0.72,
        }

        with patch.object(main, "save_uploaded_file", side_effect=self._stub_save_audio), \
            patch.object(main, "file_hashes", return_value={"sha256": "audiohash", "md5": "audio-md5"}), \
            patch.object(main, "analyse_audio_file", return_value=analysis_payload), \
            patch.object(main, "_build_audio_report_pdf", return_value=b"%PDF-audio-test"):
            response = self.client.post(
                "/analysis/audio",
                files={"file": ("sample.wav", io.BytesIO(b"fake"), "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        return response.json()["id"]

    def test_settings_round_trip_and_audit_log(self):
        response = self.client.get("/settings")
        self.assertEqual(response.status_code, 200)
        self.assertIn("pipeline", response.json())

        denied = self.client.put(
            "/settings",
            json={"pipeline": {"image_detector": "model_a"}},
        )
        self.assertEqual(denied.status_code, 403)

        updated = self.client.put(
            "/settings",
            headers={"admin-key": "test-admin-key"},
            json={"pipeline": {"image_detector": "model_a"}},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["pipeline"]["image_detector"], "model_a")

        audit = self.client.get("/audit")
        self.assertEqual(audit.status_code, 200)
        self.assertGreaterEqual(audit.json()["total"], 1)
        actions = [entry["action"] for entry in audit.json()["data"]]
        self.assertIn("settings_updated", actions)

    def test_image_analysis_detail_and_report(self):
        heatmap = self.temp_path / "heatmaps" / "gradcam.jpg"
        ela = self.temp_path / "ela" / "ela.jpg"
        noise = self.temp_path / "noise" / "noise.jpg"
        for path in (heatmap, ela, noise):
            _write_image(path)

        analysis_payload = {
            "file_integrity": {
                "hashes": {"sha256": "abc", "md5": "def"},
                "jpeg_structure": {"valid_jpeg": True},
                "hashes_match": True,
            },
            "ml": {"label": "ai"},
            "ml_prob": 0.82,
            "metadata": {"camera_make": None},
            "exif_result": {"warnings": []},
            "anomaly": {"findings": ["Missing EXIF"], "anomaly_score": 0.4},
            "qtinfo": {
                "qtables_found": True,
                "qtables": [],
                "qtables_anomaly_score": 0.1,
                "quality_estimate": 82,
                "double_compression": False,
                "inconsistency_score": 0.05,
                "combined_anomaly_score": 0.08,
            },
            "noise_info": {
                "residual_variance": 0.01,
                "spectral_flatness": 0.2,
                "noise_anomaly_score": 0.1,
                "noise_heatmap_path": str(noise),
                "inconsistency_score": 0.02,
                "combined_anomaly_score": 0.05,
            },
            "watermark_info": {
                "watermark_detected": False,
                "confidence": 0.0,
                "error": None,
            },
            "c2pa_info": {
                "has_c2pa": False,
                "signature_valid": False,
                "ai_assertions_found": [],
                "tools_detected": [],
                "edit_actions": [],
                "digital_source_types": [],
                "software_agents": [],
                "overall_c2pa_score": 0.0,
                "errors": [],
            },
            "c2pa_ai_flag": False,
            "ela_info": {
                "ela_image_path": str(ela),
                "mean_error": 1.0,
                "max_error": 2.0,
                "ela_anomaly_score": 0.1,
            },
            "heatmap_path": heatmap,
            "fused": {
                "final_score": 0.77,
                "classification": "likely_ai_generated",
                "provenance": {"fusion_mode": "rule_based_forensic_fusion"},
                "decision_path": {"decision_engine": "rule_based_forensic_fusion"},
            },
            "camera_consistency": {"consistent": False},
            "detector_info": {
                "name": "model_a",
                "display_name": "Model A",
                "model_version": "v2.1",
                "weights": {"sha256": "weights123"},
            },
        }

        with patch.object(main, "save_uploaded_file", side_effect=self._stub_save_image), \
            patch.object(main, "_ensure_image_valid", return_value=None), \
            patch.object(main, "run_full_analysis", return_value=analysis_payload), \
            patch.object(main, "_build_image_report_pdf", return_value=b"%PDF-image-test"):
            response = self.client.post(
                "/analysis/image",
                files={"file": ("sample.jpg", io.BytesIO(b"fake"), "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["classification"], "likely_ai_generated")
        self.assertEqual(payload["ml_prediction"]["detector"]["name"], "model_a")
        record_id = payload["id"]

        detail = self.client.get(f"/analysis/{record_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["forensic_score_json"]["provenance"]["fusion_mode"], "rule_based_forensic_fusion")

        report = self.client.get(f"/analysis/{record_id}/report.pdf")
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.headers["content-type"], "application/pdf")

    def test_audio_analysis_detail_and_report(self):
        record_id = self._create_audio_record()

        detail = self.client.get(f"/analysis/audio/{record_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["classification"], "likely_ai_generated")
        self.assertEqual(detail.json()["audio_features"]["analysis_mode"], "waveform")

        report = self.client.get(f"/analysis/audio/{record_id}/report.pdf")
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.headers["content-type"], "application/pdf")

    def test_video_analysis_detail_and_report(self):
        frame_path = self.temp_path / "video_frames" / "frame_0.jpg"
        _write_image(frame_path)

        analysis_payload = {
            "forensic_score": 0.61,
            "classification": "uncertain",
            "frame_count": 1,
            "frames": [
                {
                    "frame_index": 0,
                    "saved_path": str(frame_path),
                    "classification": "likely_real",
                    "forensic_score": 0.21,
                    "detector_metadata": {
                        "name": "cnndetection",
                        "display_name": "CNNDetection",
                        "model_version": "cnn-blur-jpg-0.5",
                        "weights": {"sha256": "cnnweights"},
                    },
                }
            ],
            "audio_analysis": {
                "classification": "uncertain",
                "forensic_score": 0.4,
                "waveform_path": None,
            },
        }

        with patch.object(main, "save_uploaded_file", side_effect=self._stub_save_video), \
            patch.object(main, "get_video_duration_seconds", return_value=4.0), \
            patch.object(main, "file_hashes", return_value={"sha256": "videohash", "md5": "video-md5"}), \
            patch.object(main, "extract_video_metadata", return_value={"codec": "h264"}), \
            patch.object(main, "run_video_analysis", return_value=analysis_payload), \
            patch.object(main, "_build_video_report_pdf", return_value=b"%PDF-video-test"):
            response = self.client.post(
                "/analysis/video",
                files={"file": ("sample.mp4", io.BytesIO(b"fake"), "video/mp4")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["frame_count"], 1)
        self.assertEqual(payload["video_metadata"]["image_detector"]["name"], "cnndetection")
        record_id = payload["id"]

        detail = self.client.get(f"/analysis/video/{record_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["frame_count"], 1)

        report = self.client.get(f"/analysis/video/{record_id}/report.pdf")
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.headers["content-type"], "application/pdf")

    def test_unified_analysis_listing_includes_audio(self):
        self._create_audio_record()
        listing = self.client.get("/analysis", params={"media_type": "audio"})
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["data"][0]["media_type"], "audio")


if __name__ == "__main__":
    unittest.main()
