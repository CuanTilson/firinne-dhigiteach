import uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BACKEND_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploaded"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

_CHUNK_SIZE = 1024 * 1024


def save_uploaded_file(file: UploadFile, max_bytes: int | None = None) -> Path:
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

    # Save file with optional size guard
    bytes_written = 0
    with save_path.open("wb") as buffer:
        while True:
            chunk = file.file.read(_CHUNK_SIZE)
            if not chunk:
                break
            bytes_written += len(chunk)
            if max_bytes is not None and bytes_written > max_bytes:
                buffer.close()
                save_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail="Uploaded file exceeds size limit.",
                )
            buffer.write(chunk)

    return save_path
