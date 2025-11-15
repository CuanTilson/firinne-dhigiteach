from fastapi import FastAPI, UploadFile, File
from analysis.upload import save_uploaded_file
from analysis.metadata_extractor import extract_image_metadata, extract_video_metadata
from pathlib import Path

app = FastAPI(title="Fírinne Dhigiteach API")


@app.get("/")
def read_root():
    return {"message": "Fírinne Dhigiteach API is running"}


@app.post("/upload")
async def upload_media(file: UploadFile = File(...)):
    saved_path = save_uploaded_file(file)
    return {
        "status": "success",
        "filename": file.filename,
        "saved_to": str(saved_path)
    }

@app.post("/analyse")
async def analyse_media(file: UploadFile = File(...)):
    saved_path = save_uploaded_file(file)
    ext = Path(file.filename).suffix.lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        metadata = extract_image_metadata(saved_path)
    else:
        metadata = extract_video_metadata(saved_path)

    return {
        "filename": file.filename,
        "path": str(saved_path),
        "metadata": metadata,
    }