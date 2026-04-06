import io
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

from backend.analysis import metadata, upload


class UploadValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("backend/tests/.tmp_upload_validation")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.original_upload_dir = upload.UPLOAD_DIR
        upload.UPLOAD_DIR = self.temp_dir

    def tearDown(self) -> None:
        upload.UPLOAD_DIR = self.original_upload_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_uploaded_file_rejects_missing_filename(self):
        file = UploadFile(filename="", file=io.BytesIO(b"test"))
        with self.assertRaises(HTTPException) as ctx:
            upload.save_uploaded_file(file)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Missing filename", str(ctx.exception.detail))

    def test_save_uploaded_file_rejects_unsupported_extension(self):
        file = UploadFile(filename="evidence.txt", file=io.BytesIO(b"test"))
        with self.assertRaises(HTTPException) as ctx:
            upload.save_uploaded_file(file)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Unsupported file type", str(ctx.exception.detail))

    def test_save_uploaded_file_enforces_size_limit_and_cleans_partial_file(self):
        file = UploadFile(filename="evidence.jpg", file=io.BytesIO(b"abcdef"))
        with self.assertRaises(HTTPException) as ctx:
            upload.save_uploaded_file(file, max_bytes=3)
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("size limit", str(ctx.exception.detail))
        self.assertEqual(list(self.temp_dir.iterdir()), [])


class MetadataFallbackTests(unittest.TestCase):
    def test_extract_image_metadata_returns_error_on_missing_file(self):
        result = metadata.extract_image_metadata(Path("backend/tests/does_not_exist.jpg"))
        self.assertIn("error", result)
        self.assertIsInstance(result["error"], str)

    def test_extract_video_metadata_returns_error_when_mediainfo_fails(self):
        with patch.object(metadata.MediaInfo, "parse", side_effect=RuntimeError("mediainfo unavailable")):
            result = metadata.extract_video_metadata(Path("backend/tests/fake.mp4"))
        self.assertEqual(result["error"], "mediainfo unavailable")

    def test_analyse_image_metadata_handles_missing_exif(self):
        result = metadata.analyse_image_metadata({})
        self.assertGreaterEqual(result["anomaly_score"], 0.3)
        self.assertIn("Missing or minimal EXIF data", result["findings"])


if __name__ == "__main__":
    unittest.main()
