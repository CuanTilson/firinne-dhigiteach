from fastapi import FastAPI

app = FastAPI(title="Fírinne Dhigiteach API")


@app.get("/")
def read_root():
    return {"message": "Fírinne Dhigiteach API is running"}