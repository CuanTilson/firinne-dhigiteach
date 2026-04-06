import shutil
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.analysis.audio import analyse_audio_file, extract_audio_from_video


def _write_wav(path: Path, sample_rate: int = 16000, duration_seconds: float = 0.5) -> None:
    sample_count = max(1, int(sample_rate * duration_seconds))
    samples = np.zeros(sample_count, dtype=np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


class AudioFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("backend/tests/.tmp_audio_fallbacks")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_analyse_audio_file_marks_missing_ffmpeg_for_non_wav(self):
        audio_path = self.temp_dir / "sample.mp3"
        audio_path.write_bytes(b"fake-mp3")

        with patch("backend.analysis.audio.resolve_ffmpeg_path", return_value=None):
            result = analyse_audio_file(audio_path)

        self.assertEqual(result["features"]["analysis_mode"], "basic")
        self.assertIn("ffmpeg_transcode_error", result["features"])
        self.assertIn("ffmpeg was not found", result["features"]["ffmpeg_transcode_error"])
        joined_notes = " ".join(result["metadata"]["notes"])
        self.assertIn("ffmpeg was not found", joined_notes)

    def test_analyse_audio_file_handles_short_wav_with_waveform_mode(self):
        audio_path = self.temp_dir / "short.wav"
        waveform_path = self.temp_dir / "waveform.png"
        spectrogram_path = self.temp_dir / "spectrogram.png"
        _write_wav(audio_path, duration_seconds=0.25)

        result = analyse_audio_file(audio_path, waveform_path, spectrogram_path)

        self.assertEqual(result["features"]["analysis_mode"], "waveform")
        self.assertIsNotNone(result["features"]["waveform_path"])
        self.assertTrue(waveform_path.exists())
        self.assertIn("Very short audio duration limits analysis reliability.", result["findings"])

    def test_extract_audio_from_video_reports_missing_ffmpeg(self):
        video_path = self.temp_dir / "sample.mp4"
        output_path = self.temp_dir / "extracted.wav"
        video_path.write_bytes(b"fake-video")

        with patch("backend.analysis.audio.resolve_ffmpeg_path", return_value=None):
            result = extract_audio_from_video(video_path, output_path)

        self.assertFalse(result["ok"])
        self.assertIsNone(result["output_path"])
        self.assertIn("ffmpeg was not found", result["error"])


if __name__ == "__main__":
    unittest.main()
