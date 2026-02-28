# Firinne Dhigiteach

A legal-oriented prototype for assessing whether digital media is authentic, AI-generated, or modified.

The system currently supports:
- image analysis with explainable forensic outputs
- video analysis with sampled frame review
- backend audio-analysis scaffolding for standalone audio
- case history, audit log, settings, and print/report views
- reproducible Model A training from CSV manifests

The project is being delivered in weekly stages for a final-year dissertation, with traceability and reproducibility treated as first-class requirements.

## Current Status

Implemented now:
- FastAPI backend for image and video analysis
- FastAPI backend audio-analysis scaffold and audio report endpoint
- React frontend for upload, review, history, and reporting
- forensic signals including GradCAM, ELA, metadata checks, C2PA, JPEG analysis, and noise residuals
- Week 2 data pipeline for reproducible image-model training
- Week 3 Model A baseline training scaffold and first baseline run

Planned next:
- audio extraction from uploaded video
- explicit decision-path presentation for rule-based vs learned-fusion outputs
- broader evaluation and final report packaging

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
- The current baseline shows strong in-domain performance and weak external generalization, which is an important project finding rather than a failure.
