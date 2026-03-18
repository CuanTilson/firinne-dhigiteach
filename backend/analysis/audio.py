from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _safe_float(value: float | int | None, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_audio_metadata(audio_path: Path) -> dict:
    metadata = {
        "filename": audio_path.name,
        "extension": audio_path.suffix.lower(),
        "size_bytes": audio_path.stat().st_size if audio_path.exists() else 0,
        "container_supported": False,
        "parse_method": "basic_file_info",
        "duration_seconds": None,
        "sample_rate_hz": None,
        "channels": None,
        "sample_width_bytes": None,
        "notes": [],
    }

    if audio_path.suffix.lower() != ".wav":
        metadata["notes"].append(
            "Detailed waveform metadata is currently implemented for WAV files only."
        )
        return metadata

    try:
        with contextlib.closing(wave.open(str(audio_path), "rb")) as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            duration = frames / float(sample_rate) if sample_rate else 0.0
            metadata.update(
                {
                    "container_supported": True,
                    "parse_method": "wave",
                    "duration_seconds": duration,
                    "sample_rate_hz": sample_rate,
                    "channels": channels,
                    "sample_width_bytes": sample_width,
                    "frame_count": frames,
                }
            )
    except wave.Error as exc:
        metadata["notes"].append(f"Wave parsing failed: {exc}")

    return metadata


def _dtype_for_sample_width(sample_width: int):
    if sample_width == 1:
        return np.uint8
    if sample_width == 2:
        return np.int16
    if sample_width == 4:
        return np.int32
    return None


def _load_wav_samples(audio_path: Path) -> tuple[np.ndarray | None, dict]:
    metadata = extract_audio_metadata(audio_path)
    if metadata.get("parse_method") != "wave":
        return None, metadata

    try:
        with contextlib.closing(wave.open(str(audio_path), "rb")) as wav_file:
            raw = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
    except wave.Error as exc:
        metadata["notes"].append(f"Wave sample load failed: {exc}")
        return None, metadata

    dtype = _dtype_for_sample_width(sample_width)
    if dtype is None:
        metadata["notes"].append(
            f"Unsupported WAV sample width for deep analysis: {sample_width} bytes."
        )
        return None, metadata

    samples = np.frombuffer(raw, dtype=dtype)
    if dtype == np.uint8:
        samples = samples.astype(np.float32)
        samples = (samples - 128.0) / 128.0
    else:
        max_abs = float(2 ** (8 * sample_width - 1))
        samples = samples.astype(np.float32) / max_abs

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, metadata


def generate_waveform_image(audio_path: Path, output_path: Path) -> str | None:
    samples, metadata = _load_wav_samples(audio_path)
    if samples is None or samples.size == 0:
        return None

    width = 1200
    height = 320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline="#cbd5e1")
    mid_y = height // 2
    draw.line((0, mid_y, width, mid_y), fill="#94a3b8", width=1)

    bucket_size = max(1, samples.size // width)
    points = []
    for x in range(width):
        start = x * bucket_size
        end = min(samples.size, start + bucket_size)
        if start >= samples.size:
            break
        segment = samples[start:end]
        amplitude = float(np.max(np.abs(segment))) if segment.size else 0.0
        y = int(mid_y - amplitude * (height * 0.42))
        points.append((x, y))
        points.append((x, height - y))

    for x, y in points[::2]:
        mirror_y = height - y
        draw.line((x, y, x, mirror_y), fill="#0f766e", width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG")
    return str(output_path)


def generate_spectrogram_image(audio_path: Path, output_path: Path) -> str | None:
    samples, metadata = _load_wav_samples(audio_path)
    sample_rate = int((metadata or {}).get("sample_rate_hz") or 0)
    if samples is None or samples.size < 512 or sample_rate <= 0:
        return None

    fft_size = 512
    hop = 256
    if samples.size < fft_size:
        return None

    frame_count = 1 + max(0, (samples.size - fft_size) // hop)
    if frame_count <= 0:
        return None

    window = np.hanning(fft_size).astype(np.float32)
    spectrogram: list[np.ndarray] = []
    for frame_index in range(frame_count):
        start = frame_index * hop
        segment = samples[start : start + fft_size]
        if segment.size < fft_size:
            break
        magnitude = np.abs(np.fft.rfft(segment * window))
        spectrogram.append(magnitude)

    if not spectrogram:
        return None

    spec = np.stack(spectrogram, axis=1)
    spec = np.log1p(spec)
    spec -= float(spec.min())
    max_value = float(spec.max())
    if max_value > 0:
        spec /= max_value

    spec = np.flipud(spec)
    image_array = np.clip(spec * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L").convert("RGB")
    image = image.resize((1200, 320), Image.Resampling.BILINEAR)

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline="#cbd5e1")
    draw.text((12, 10), "Spectrogram", fill="#e2e8f0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG")
    return str(output_path)


def _segment_feature_summary(samples: np.ndarray, sample_rate: int) -> dict | None:
    if samples.size == 0 or sample_rate <= 0:
        return None

    segment_length = max(sample_rate // 2, 1)
    segment_count = samples.size // segment_length
    if segment_count <= 0:
        return None

    rms_values: list[float] = []
    zcr_values: list[float] = []
    flatness_values: list[float] = []
    centroid_values: list[float] = []

    for segment_index in range(segment_count):
        start = segment_index * segment_length
        segment = samples[start : start + segment_length]
        if segment.size < 32:
            continue

        rms_values.append(float(np.sqrt(np.mean(np.square(segment)))))
        zcr_values.append(
            float(np.mean(np.abs(np.diff(np.signbit(segment).astype(np.int8)))))
        )

        spectrum = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
        if spectrum.size:
            safe_spectrum = spectrum + 1e-12
            flatness_values.append(
                float(np.exp(np.mean(np.log(safe_spectrum))) / np.mean(safe_spectrum))
            )
            denom = float(np.sum(spectrum))
            centroid_values.append(
                float(np.sum(freqs * spectrum) / denom) if denom > 0 else 0.0
            )

    if not rms_values:
        return None

    def summarize(values: list[float]) -> dict:
        array = np.asarray(values, dtype=np.float32)
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    return {
        "segment_duration_seconds": 0.5,
        "segment_count": len(rms_values),
        "rms_level": summarize(rms_values),
        "zero_crossing_rate": summarize(zcr_values),
        "spectral_flatness": summarize(flatness_values),
        "spectral_centroid_hz": summarize(centroid_values),
    }


def _dynamic_range_db(samples: np.ndarray) -> float | None:
    if samples.size == 0:
        return None
    magnitudes = np.abs(samples)
    low = float(np.percentile(magnitudes, 10))
    high = float(np.percentile(magnitudes, 95))
    if high <= 1e-8:
        return None
    return float(20.0 * np.log10((high + 1e-8) / (low + 1e-8)))


def _repetition_score(samples: np.ndarray, sample_rate: int) -> float | None:
    if samples.size == 0 or sample_rate <= 0:
        return None

    segment_length = max(sample_rate // 2, 1)
    segment_count = samples.size // segment_length
    if segment_count < 3:
        return None

    signatures: list[np.ndarray] = []
    for segment_index in range(segment_count):
        start = segment_index * segment_length
        segment = samples[start : start + segment_length]
        if segment.size < 128:
            continue
        spectrum = np.abs(np.fft.rfft(segment[: min(segment.size, 4096)]))
        if spectrum.size < 8:
            continue
        spectrum = spectrum / (np.linalg.norm(spectrum) + 1e-8)
        signatures.append(spectrum.astype(np.float32))

    if len(signatures) < 3:
        return None

    similarities: list[float] = []
    for i in range(len(signatures) - 1):
        a = signatures[i]
        b = signatures[i + 1]
        similarities.append(float(np.clip(np.dot(a, b), 0.0, 1.0)))

    if not similarities:
        return None
    return float(np.mean(similarities))


def _transcode_audio_to_wav(
    audio_path: Path,
    output_path: Path,
    ffmpeg_path: str,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "output_path": None,
            "error": f"ffmpeg execution failed: {exc}",
        }

    if completed.returncode != 0 or not output_path.exists():
        return {
            "ok": False,
            "output_path": None,
            "error": (completed.stderr or completed.stdout or "ffmpeg failed").strip(),
        }

    return {
        "ok": True,
        "output_path": str(output_path),
        "error": None,
    }


def analyse_audio_file(
    audio_path: Path,
    waveform_output_path: Path | None = None,
    spectrogram_output_path: Path | None = None,
    configured_ffmpeg_path: str | None = None,
) -> dict:
    metadata = extract_audio_metadata(audio_path)
    analysis_input_path = audio_path
    transcoded_tmp: Path | None = None
    transcode_error: str | None = None

    if audio_path.suffix.lower() != ".wav":
        ffmpeg_path = resolve_ffmpeg_path(configured_ffmpeg_path)
        if ffmpeg_path:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                transcoded_tmp = Path(tmp.name)
            transcoded = _transcode_audio_to_wav(audio_path, transcoded_tmp, ffmpeg_path)
            if transcoded["ok"]:
                analysis_input_path = transcoded_tmp
                metadata["notes"].append(
                    "Non-WAV input was transcoded to mono 16 kHz WAV for waveform analysis."
                )
                metadata["container_supported"] = True
                metadata["parse_method"] = "ffmpeg_transcode"
            else:
                transcode_error = str(transcoded["error"] or "ffmpeg transcode failed")
                metadata["notes"].append(
                    f"ffmpeg transcode was unavailable for deep analysis: {transcode_error}"
                )

    samples, deep_metadata = _load_wav_samples(analysis_input_path)
    if deep_metadata:
        if metadata.get("parse_method") == "ffmpeg_transcode":
            for key in (
                "duration_seconds",
                "sample_rate_hz",
                "channels",
                "sample_width_bytes",
                "frame_count",
            ):
                metadata[key] = deep_metadata.get(key)
        else:
            metadata = deep_metadata

    features = {
        "rms_level": None,
        "peak_level": None,
        "clipping_ratio": None,
        "silence_ratio": None,
        "dynamic_range_db": None,
        "zero_crossing_rate": None,
        "crest_factor": None,
        "spectral_centroid_hz": None,
        "dominant_frequency_hz": None,
        "spectral_flatness": None,
        "repetition_score": None,
        "waveform_path": None,
        "spectrogram_path": None,
        "segment_summary": None,
        "analysis_mode": "basic",
        "transcoded_for_analysis": analysis_input_path != audio_path,
    }
    findings: list[str] = []
    score = 0.25

    try:
        if samples is not None and samples.size > 0:
            rms = float(np.sqrt(np.mean(np.square(samples))))
            peak = float(np.max(np.abs(samples)))
            clipping_ratio = float(np.mean(np.abs(samples) >= 0.99))
            silence_ratio = float(np.mean(np.abs(samples) <= 0.002))
            zero_crossing_rate = float(
                np.mean(np.abs(np.diff(np.signbit(samples).astype(np.int8))))
            )
            crest_factor = float(peak / rms) if rms > 1e-8 else None
            sample_rate = int(metadata.get("sample_rate_hz") or 0)
            dynamic_range_db = _dynamic_range_db(samples)
            repetition_score = _repetition_score(samples, sample_rate)
            centroid = None
            dominant_frequency = None
            flatness = None
            if sample_rate > 0:
                window = samples[: min(samples.size, 16384)]
                if window.size > 32:
                    spectrum = np.abs(np.fft.rfft(window))
                    freqs = np.fft.rfftfreq(window.size, d=1.0 / sample_rate)
                    denom = float(np.sum(spectrum))
                    centroid = float(np.sum(freqs * spectrum) / denom) if denom > 0 else 0.0
                    if spectrum.size > 1:
                        dominant_idx = int(np.argmax(spectrum[1:]) + 1)
                        dominant_frequency = float(freqs[dominant_idx])
                    safe_spectrum = spectrum + 1e-12
                    flatness = float(
                        np.exp(np.mean(np.log(safe_spectrum))) / np.mean(safe_spectrum)
                    )

            features.update(
                {
                    "rms_level": rms,
                    "peak_level": peak,
                    "clipping_ratio": clipping_ratio,
                    "silence_ratio": silence_ratio,
                    "dynamic_range_db": dynamic_range_db,
                    "zero_crossing_rate": zero_crossing_rate,
                    "crest_factor": crest_factor,
                    "spectral_centroid_hz": centroid,
                    "dominant_frequency_hz": dominant_frequency,
                    "spectral_flatness": flatness,
                    "repetition_score": repetition_score,
                    "analysis_mode": "waveform"
                    if analysis_input_path == audio_path
                    else "transcoded_waveform",
                }
            )

            if waveform_output_path is not None:
                features["waveform_path"] = generate_waveform_image(
                    analysis_input_path, waveform_output_path
                )
            if spectrogram_output_path is not None:
                features["spectrogram_path"] = generate_spectrogram_image(
                    analysis_input_path, spectrogram_output_path
                )
            features["segment_summary"] = _segment_feature_summary(samples, sample_rate)

            if clipping_ratio > 0.02:
                findings.append("High clipping ratio may indicate aggressive editing or poor capture quality.")
                score += 0.25
            if silence_ratio > 0.8:
                findings.append("Large silent proportion reduces evidential quality.")
                score += 0.15
            if rms < 0.01:
                findings.append("Very low overall signal level limits interpretation confidence.")
                score += 0.10
            if crest_factor is not None and crest_factor > 8.0:
                findings.append("High crest factor indicates sharp transients or heavy dynamic shaping.")
                score += 0.10
            if dynamic_range_db is not None and dynamic_range_db < 12.0:
                findings.append("Narrow dynamic range suggests strong compression, leveling, or synthetic regularity.")
                score += 0.08
            if flatness is not None and flatness > 0.45:
                findings.append("Noise-like spectral flatness suggests strong synthesis, compression, or degraded capture quality.")
                score += 0.10
            if zero_crossing_rate > 0.25:
                findings.append("High zero-crossing rate indicates a noisy or strongly high-frequency signal profile.")
                score += 0.08
            if repetition_score is not None and repetition_score > 0.96:
                findings.append("High adjacent-segment similarity suggests repetitive or looped signal structure.")
                score += 0.07
            segment_summary = features.get("segment_summary")
            if isinstance(segment_summary, dict):
                rms_summary = segment_summary.get("rms_level")
                zcr_summary = segment_summary.get("zero_crossing_rate")
                if isinstance(rms_summary, dict) and _safe_float(rms_summary.get("std")) > 0.12:
                    findings.append("Segment-to-segment energy varies strongly across the recording.")
                    score += 0.06
                if isinstance(zcr_summary, dict) and _safe_float(zcr_summary.get("std")) > 0.08:
                    findings.append("Segmented zero-crossing rate varies strongly, suggesting unstable signal texture.")
                    score += 0.05
            if sample_rate and sample_rate < 16000:
                findings.append("Low sample rate limits spectral detail and evidential quality.")
                score += 0.05
            if metadata.get("duration_seconds") and _safe_float(metadata["duration_seconds"]) < 1.0:
                findings.append("Very short audio duration limits analysis reliability.")
                score += 0.10
        else:
            findings.append(
                "Deep waveform analysis is currently unavailable for this audio container; metadata-only triage was used."
            )
            score += 0.15
    finally:
        if transcoded_tmp is not None:
            transcoded_tmp.unlink(missing_ok=True)

    if transcode_error and features["analysis_mode"] == "basic":
        features["ffmpeg_transcode_error"] = transcode_error

    score = max(0.0, min(score, 1.0))
    if score >= 0.7:
        classification = "likely_ai_generated"
    elif score <= 0.3:
        classification = "likely_real"
    else:
        classification = "uncertain"

    return {
        "metadata": metadata,
        "features": features,
        "findings": findings,
        "forensic_score": score,
        "classification": classification,
    }


def extract_audio_from_video(
    video_path: Path,
    output_path: Path,
    configured_ffmpeg_path: str | None = None,
) -> dict:
    ffmpeg_path = resolve_ffmpeg_path(configured_ffmpeg_path)
    if not ffmpeg_path:
        return {
            "ok": False,
            "output_path": None,
            "error": "ffmpeg was not found. Configure an explicit ffmpeg path in settings or install it on PATH.",
        }

    return _extract_audio_from_video(video_path, output_path, ffmpeg_path)


def resolve_ffmpeg_path(configured_path: str | None = None) -> str | None:
    candidates: list[str] = []
    if configured_path:
        candidates.append(configured_path)

    env_path = os.getenv("FD_FFMPEG_PATH")
    if env_path:
        candidates.append(env_path)

    path_lookup = shutil.which("ffmpeg")
    if path_lookup:
        candidates.append(path_lookup)

    local_candidates = [
        Path("tools/ffmpeg/bin/ffmpeg.exe"),
        Path("backend/tools/ffmpeg/bin/ffmpeg.exe"),
        Path.home() / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(os.getenv("ProgramFiles", "")) / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(os.getenv("ProgramFiles(x86)", "")) / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(os.getenv("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/tools/ffmpeg/bin/ffmpeg.exe"),
    ]
    candidates.extend(str(path) for path in local_candidates if str(path).strip())

    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(Path(candidate)).strip().strip('"')
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if Path(normalized).is_file():
            return normalized
        resolved = shutil.which(normalized)
        if resolved:
            return resolved
    return None


def _extract_audio_from_video(video_path: Path, output_path: Path, ffmpeg_path: str) -> dict:
    if not ffmpeg_path:
        return {
            "ok": False,
            "output_path": None,
            "error": "ffmpeg was not found.",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "output_path": None,
            "error": f"ffmpeg execution failed: {exc}",
        }

    if completed.returncode != 0 or not output_path.exists():
        return {
            "ok": False,
            "output_path": None,
            "error": (completed.stderr or completed.stdout or "ffmpeg failed").strip(),
        }

    return {
        "ok": True,
        "output_path": str(output_path),
        "error": None,
    }
