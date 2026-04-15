# Fírinne Dhigiteach

Fírinne Dhigiteach is a multimodal forensic decision-support prototype for analysing images, video, and audio. It combines a self-trained image classifier with forensic signals such as metadata checks, C2PA provenance parsing, JPEG structure analysis, ELA, noise residual analysis, audio signal diagnostics, hashing, audit logging, and report generation.

The system is intended as decision-support evidence, not as a definitive legal determination.

## Main Features

- Image analysis with ML probability, forensic fusion, Grad-CAM, ELA, metadata, JPEG, C2PA, noise, and watermark checks.
- Video analysis through sampled frame analysis and optional extracted-audio triage.
- Standalone audio analysis with waveform, spectrogram, segmented signal features, and FFmpeg-based decoding when available.
- Case history, audit log, settings, applied-settings snapshots, and print/PDF-style report views.
- Runtime use of the self-trained `Model A v2.1` detector when the model artefacts are present.
- Backend integration tests and consolidated Chapter 6 evaluation artefacts.

## Repository Layout

```text
backend/                 FastAPI API, analysis pipeline, database models, tests, training tools
frontend/                React/Vite user interface
data/                    Dataset manifest scripts and manifest outputs
artifacts/               Local model/evaluation artefacts; not committed by default
docs/                    Local dissertation/project notes; not required to run the app
Dockerfile.backend       Backend container definition
Dockerfile.frontend      Frontend container definition
docker-compose.yml       Local Docker Compose setup
```

## Required Model Artefacts

The default image detector is the self-trained `Model A v2.1`. For full functionality, place the model artefacts at:

```text
artifacts/model_a_v2_1_gpu/model_a_best.pt
artifacts/model_a_v2_1_gpu/run_manifest.json
```

Useful evaluation outputs are stored under:

```text
artifacts/model_a_v2_1_gpu/exports/
artifacts/chapter6_evaluation/
```

If these artefacts are missing, image analysis will not have an available detector.

## Local Development

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd ..
.\backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Backend URL:

```text
http://localhost:8000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

Run the production frontend build from the `frontend/` directory:

```powershell
npm run build
```

## Docker

Start both services:

```powershell
docker compose up --build
```

Container URLs:

```text
frontend: http://localhost:5173
backend:  http://localhost:8000
```

Docker Compose mounts:

```text
./backend/storage
./backend/database
./artifacts
```

The model artefacts are not baked into the Docker image. They are expected in the local `artifacts/` folder and mounted into the backend container.

## Configuration

Backend configuration is read from `backend/.env` when present.

Common backend variables:

```text
FD_ADMIN_KEY
FD_API_KEY
FD_CORS_ORIGINS
FD_IMAGE_DETECTOR
FD_MODEL_A_WEIGHTS
FD_MODEL_A_RUN_MANIFEST
FD_FFMPEG_PATH
FD_MAX_IMAGE_MB
FD_MAX_VIDEO_MB
FD_MAX_AUDIO_MB
```

Frontend configuration is read from `frontend/.env` when present.

Common frontend variables:

```text
VITE_API_BASE_URL
VITE_API_KEY
VITE_ADMIN_KEY
```

## Audio and FFmpeg

FFmpeg is used for audio/video decoding and extraction.

- `.wav` files can be analysed without FFmpeg.
- `.mp3`, `.m4a`, and `.flac` get deeper waveform analysis when FFmpeg is available.
- Video extracted-audio analysis requires FFmpeg.

Recommended Windows path if auto-discovery fails:

```text
C:\ffmpeg\bin\ffmpeg.exe
```

This can be set through the Settings page or with `FD_FFMPEG_PATH`.

## Tests

Run backend tests from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m unittest backend.tests.test_upload_and_metadata backend.tests.test_audio_fallbacks backend.tests.test_api_integration
```

Run frontend type/build checks:

```powershell
cd frontend
npm run build
```

## Submission Notes

For a Brightspace code ZIP, include source code and the final `artifacts/model_a_v2_1_gpu/` folder. Do not include virtual environments, `node_modules`, local databases, runtime storage files, or raw datasets.
