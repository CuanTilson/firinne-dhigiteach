# Fírinne Dhigiteach – Deepfake & Digital Media Forensics

### Prototype System for Image Authenticity Analysis

This project is a forensic analysis pipeline for detecting AI-generated images using a combination of:

- CNN‐based deepfake detection
- GradCAM visual explainability
- Metadata & EXIF forensics
- JPEG structure and Q-table analysis
- Error Level Analysis (ELA)
- Noise residual analysis
- Stable Diffusion invisible watermark detection
- C2PA provenance verification
- A forensic fusion algorithm combining all signals
- A React-based frontend for interactive visualisation

---

# 1. Project Structure

```
backend/
  ├── analysis/
  ├── models/
  ├── explainability/
  ├── database/
  ├── storage/
  │     ├── uploaded/
  │     ├── ela/
  │     ├── heatmaps/
  │     └── thumbnails/
  ├── main.py
  └── ...

frontend/
  └── (Vite + React + Tailwind)
vendor/
  └── CNNDetection/ (submodule)
```

---

# 2. Requirements

### Backend (Python)

- Python 3.10+
- pip + venv
- PyTorch (CPU is fine for prototype)
- FastAPI
- SQLite (already included)

### Frontend (Node)

- Node.js 18+
- npm or yarn

---

# 3. Backend Setup (Python + FastAPI)

### 3.1 Create and activate the virtual environment

**macOS/Linux**

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
```

---

### 3.2 Install backend dependencies

```bash
pip install -r requirements.txt
```

If your project has no `requirements.txt` yet, generate one:

```bash
pip freeze > requirements.txt
```

---

### 3.3 Ensure storage folders exist

These are automatically created by `main.py`, but you can check:

```
backend/storage/uploaded
backend/storage/ela
backend/storage/heatmaps
backend/storage/thumbnails
```

---

### 3.4 Start the backend server

Development mode with auto-reload:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend docs live at:

```
http://localhost:8000/docs
http://localhost:8000/redoc
```

---

# 4. CNNDetection Model Setup (Vendor Submodule)

You included CNNDetection as a **git submodule** under `vendor/`.
https://github.com/PeterWang512/CNNDetection

### 4.1 Initialise submodules

If you freshly cloned the repo:

```bash
git submodule update --init --recursive
```

If needed, pull nested updates later:

```bash
git submodule update --recursive --remote
```

---

### 4.2 Download CNNDetection weights

From project root:

```bash
cd vendor/CNNDetection/weights
```

Download the pretrained weights:

```bash
wget https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=0 -O blur_jpg_prob0.5.pth
wget https://www.dropbox.com/s/h7tkpcgiwuftb6g/blur_jpg_prob0.1.pth?dl=0 -O blur_jpg_prob0.1.pth
```

Or if using PowerShell (Windows):

```powershell
Invoke-WebRequest "https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=0" -OutFile blur_jpg_prob0.5.pth
Invoke-WebRequest "https://www.dropbox.com/s/h7tkpcgiwuftb6g/blur_jpg_prob0.1.pth?dl=0" -OutFile blur_jpg_prob0.1.pth
```

---

### 4.3 Backend Model Path

Your backend loads:

```python
WEIGHTS = Path("vendor/CNNDetection/weights/blur_jpg_prob0.5.pth")
```

So ensure:

```
vendor/CNNDetection/weights/blur_jpg_prob0.5.pth
```

exists.

---

# 5. Frontend Setup (React + Vite)

### 5.1 Install dependencies

```bash
cd frontend
npm install
```

Or:

```bash
yarn install
```

---

### 5.2 Start the dev server

```bash
npm run dev
```

Frontend runs on:

```
http://localhost:5173/
```

---

### 5.3 API configuration

The frontend reads the backend URL from:

```
frontend/src/constants.ts
```

Default:

```ts
export const API_BASE_URL = "http://localhost:8000";
```

Change only if backend runs elsewhere.

---

# 6. Running the Full System

### Backend (terminal 1)

```bash
cd backend
source venv/bin/activate   # or venv\Scripts\activate on Windows
uvicorn backend.main:app --reload
```

### Frontend (terminal 2)

```bash
cd frontend
npm run dev
```

Then open:

```
http://localhost:5173/
```

---

# 7. Features Implemented

- CNN-based real-vs-AI classification
- GradCAM heatmap generation
- Full EXIF + metadata inspection
- JPEG structure analysis (SOI/EOI markers, APP segments, double compression)
- JPEG quantisation table anomaly scoring
- Noise residual analysis (variance + spectral flatness)
- Stable Diffusion watermark decoding
- Error Level Analysis (ELA) with preview heatmap
- C2PA provenance extraction + AI-assertion detection
- Forensic fusion scoring
- SQLite database of past analyses
- React dashboard display with switchable heatmaps
- Thumbnail history view
- Case inspector with metadata and C2PA tabs

---

# 8. Folder Overview

### Backend

```
backend/main.py                  → main FastAPI app
backend/analysis/                → forensic modules
backend/models/                  → CNN model loader
backend/explainability/          → GradCAM generator
backend/database/                → DB and ORM
backend/storage/                 → uploaded files + output heatmaps
```

### Frontend

```
frontend/src/
  components/
  pages/
  services/api.ts
  constants.ts
  types.ts
```

### Vendor

```
vendor/CNNDetection/             → third-party detection model
```

---

# 9. Useful Commands

### Regenerate DB (fresh start)

```bash
rm backend/database/forensics.db
```

### Test backend manually

```bash
curl -X POST -F "file=@test.jpg" http://localhost:8000/analysis/image
```

### Check submodules

```bash
git submodule status
```
