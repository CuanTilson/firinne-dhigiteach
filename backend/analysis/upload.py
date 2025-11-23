import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException

UPLOAD_DIR = Path("backend/storage/uploaded")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".mkv"}


def save_uploaded_file(file: UploadFile) -> Path:
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed types: {ALLOWED_EXTENSIONS}",
        )

    # Create a save path
    save_path = UPLOAD_DIR / f"{file.filename}"

    # Save file
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return save_path
