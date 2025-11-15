from fastapi import FastAPI, UploadFile, File
from analysis.upload import save_uploaded_file

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
