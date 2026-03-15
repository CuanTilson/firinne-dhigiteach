# Firinne Dhigiteach

A legal-oriented prototype for assessing whether digital media is authentic, AI-generated, or modified.

The system currently supports:
- image analysis with explainable forensic outputs
- video analysis with sampled frame review
- backend audio-analysis scaffolding for standalone audio
- case history, audit log, settings, and print/report views
- reproducible Model A training from CSV manifests
- runtime-selectable comparison detector path for the self-trained `Model A`

The project is being delivered in weekly stages for a final-year dissertation, with traceability and reproducibility treated as first-class requirements.

## Current Status

Implemented now:
- FastAPI backend for image and video analysis
- FastAPI backend audio-analysis scaffold and audio report endpoint
- React frontend for upload, review, history, and reporting
- forensic signals including GradCAM, ELA, metadata checks, C2PA, JPEG analysis, and noise residuals
- Week 2 data pipeline for reproducible image-model training
- Week 3 Model A baseline training scaffold and first baseline run
- Week 5 corrected broader-data `Model A v2.1` experiment and coverage audit

Planned next:
- broader evaluation and final report packaging
- backend and route-level testing hardening
- final report and PDF presentation polish

## Repository Layout

```text
backend/                 FastAPI app, forensic analysis, training scaffolds
frontend/                React frontend
data/                    manifests, data scripts, and data workflow docs
artifacts/               local training outputs (not committed)
docs/                    project documentation by topic and week
vendor/CNNDetection/     third-party model dependency
```

## Quick Start

### Backend

```powershell
cd backend
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
cd ..
./backend/.venv/Scripts/python.exe -m uvicorn backend.main:app --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend default URL:
- `http://localhost:5173`

Backend default URL:
- `http://localhost:8000`

## Configuration

Backend configuration is driven by `backend/.env`.
Key variables:
- `FD_ADMIN_KEY`
- `FD_API_KEY`
- `FD_CORS_ORIGINS`
- `FD_RATE_LIMIT_PER_MINUTE`
- `FD_MAX_IMAGE_MB`
- `FD_MAX_VIDEO_MB`
- `FD_MAX_UPLOAD_MB`
- `FD_RETENTION_DAYS`
- `FD_RETENTION_INTERVAL_HOURS`

Frontend configuration is driven by `frontend/.env`.
Key variables:
- `VITE_API_BASE_URL`
- `VITE_API_KEY`
- `VITE_ADMIN_KEY`

## Model Dependency

The backend still depends on the CNNDetection vendor weights for the current forensic image pipeline.
Expected path:
- `vendor/CNNDetection/weights/blur_jpg_prob0.5.pth`

The runtime comparison path for the self-trained detector now defaults to the corrected broader-data run:
- `artifacts/model_a_v2_1_gpu/model_a_best.pt`
- `artifacts/model_a_v2_1_gpu/run_manifest.json`

This does not replace the production rule-based path. It uses the stronger corrected `Model A`
comparison checkpoint while keeping the main production decision path unchanged.

## Audio and FFmpeg

`ffmpeg` is used for media decoding and extraction, not for the core scoring logic.

Current behavior:
- `.wav` uploads:
  - deep waveform analysis without `ffmpeg`
- `.mp3`, `.m4a`, `.flac` uploads:
  - with `ffmpeg`: decoded to mono 16 kHz WAV and analysed with the deeper waveform path
  - without `ffmpeg`: metadata-first triage only
- video uploads:
  - extracted-audio analysis requires `ffmpeg`

The backend resolves `ffmpeg` in this order:
1. settings page override
2. `FD_FFMPEG_PATH`
3. system `PATH`
4. common Windows install locations

If auto-discovery fails, set the full `ffmpeg.exe` path in the settings page.

If you clone the project fresh, initialize submodules first:

```powershell
git submodule update --init --recursive
```

## Documentation Map

Start here for project docs:
- `docs/README.md`

Important supporting docs:
- `docs/project-status.md` - current implementation and research status
- `data/README.md` - dataset storage, manifest generation, validation workflow
- `backend/models/training/README.md` - Model A training workflow
- `docs/Week 1/week1-scope-freeze.md` - frozen project scope
- `docs/Week 1/week1-evaluation-protocol.md` - evaluation protocol
- `docs/Week 2/week2-summary.md` - data preparation completion summary
- `docs/Week 3/week3-model-a-baseline.md` - first baseline training result
- `docs/Week 3/week3-evaluation-summary.md` - consolidated Week 3 evaluation
- `docs/Week 4/week4-audio-backend-scaffold.md` - Week 4 audio backend increment

## Notes

- Raw datasets should stay outside the repository.
- Training artifacts should be written to `artifacts/` and not committed.
- The original Model A baseline showed strong in-domain performance and weak external generalization.
- The corrected broader-data `Model A v2.1` run improved external performance substantially and
  is now the preferred self-trained comparison checkpoint.
