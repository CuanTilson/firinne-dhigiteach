import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException

UPLOAD_DIR = Path("backend/storage/uploaded")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".mkv"}


def save_uploaded_file(file: UploadFile) -> Path:
    # Validate extension
    original_name = Path(file.filename or "").name
    if not original_name:
        raise HTTPException(status_code=400, detail="Missing filename.")

    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed types: {ALLOWED_EXTENSIONS}",
        )

    # Create a save path
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"

    # Save file
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return save_path
