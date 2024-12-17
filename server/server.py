from typing import List

from fastapi import FastAPI, UploadFile, File, Form
import os
from pathlib import Path
import subprocess
import shutil
import uuid

from fastapi.responses import RedirectResponse

server = FastAPI()

DATA_DIR = os.getenv("DATA_DIR", "../data/")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


@server.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@server.get("/health")
def health():
    return {"status": "ok"}

@server.get("/model_version")
def get_model_version():
    version_file = Path(DATA_DIR) / "model_version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            return {"version": f.read().strip()}
    return {"version": "0"}

@server.get("/model")
def get_model():
    return {
        "version": get_model_version()["version"],
        "model": "model"
    }

def train_model() -> str:
    """Simulates training a model and saving it as ONNX."""
    version = str(uuid.uuid4())[:8]  # Generate a unique version
    models_dir = Path(DATA_DIR) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{version}.onnx"
    with open(model_path, "wb") as f:
        f.write(b"This is a placeholder for ONNX model data.")

    # Update the model version
    version_file = Path(DATA_DIR) / "model_version.txt"
    with open(version_file, "w") as f:
        f.write(version)

    return version

def git_add_commit_push(model_version: str, branch: str):
    """Adds, commits, and pushes the new model to a specific branch using SSH."""
    repo_dir = Path(DATA_DIR)

    # Ensure SSH is used for Git
    ssh_url = "git@github.com:user/repo.git"  # Replace with your repo's SSH URL
    subprocess.run(["git", "remote", "set-url", "origin", ssh_url], cwd=repo_dir, check=True)

    # Git add
    subprocess.run(["git", "add", "models/"], cwd=repo_dir, check=True)

    # Git commit
    commit_message = f"Add new model version {model_version}"
    subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_dir, check=True)

    # Git push
    subprocess.run(["git", "push", "origin", branch], cwd=repo_dir, check=True)

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

    # Train the model
    model_version = train_model()

    # Add, commit, and push the new model
    branch_name = "new-model-updates"  # Specify the branch
    try:
        git_add_commit_push(model_version, branch_name)
    except subprocess.CalledProcessError as e:
        return {
            "message": f"Person '{name}' added with {len(photos)} photos.",
            "photos_saved": [photo.filename for photo in photos],
            "error": f"Failed to push model to Git: {e}"
        }

    return {
        "message": f"Person '{name}' added with {len(photos)} photos.",
        "photos_saved": [photo.filename for photo in photos],
        "model_version": model_version,
        "git_status": "Pushed to branch 'new-model-updates'"
    }
