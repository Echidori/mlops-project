from typing import List

from fastapi import FastAPI, UploadFile, File, Form
import os
from pathlib import Path

server = FastAPI()

DATA_DIR = os.getenv("DATA_DIR", "../data/")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

@server.get("/health")
def health():
    return {"status": "ok"}

@server.get("/model_version")
def get_model_version():
    return {
        "version": "0"
    }

@server.get("/model")
def get_model():
    return {
        "version": "0",
        "model": "model"
    }

@server.post("/add_person")
async def add_person(
        name: str = Form(...),
    photos: List[UploadFile] = File(...)
):
    images_dir = Path(DATA_DIR) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for photo in photos:
        file_path = images_dir / photo.filename
        with open(file_path, "wb") as f:
            f.write(await photo.read())

    return {
        "message": f"Person '{name}' added with {len(photos)} photos.",
        "photos_saved": [photo.filename for photo in photos]
    }

