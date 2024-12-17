from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import RedirectResponse

import os
from pathlib import Path
import subprocess



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

@server.get("/get_names")
def get_names():
    names_file = Path(DATA_DIR) / "names.json"
    if names_file.exists():
        with open(names_file, "r") as f:
            return {"names": f.read()}
    return {"names": "[]"}

@server.get("/get_label_map")
def get_label_map():
    label_map_file = Path(DATA_DIR) / "index_to_label.json"
    if label_map_file.exists():
        with open(label_map_file, "r") as f:
            return {"index_to_label": f.read()}
    return {"index_to_label": "{}"}


def train_model() -> str:
    try:
        result = subprocess.run(
            ["python", "../src/cnn_train.py"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)

        # Read the model version from the file
        version_file = Path("../data/model_version.txt")
        if version_file.exists():
            with open(version_file, "r") as f:
                model_version = f.read().strip()
                return model_version
        else:
            print("Model version file not found.")
            return "0"
    except subprocess.CalledProcessError as e:
        print(f"Error during model training: {e.stderr}")
        return "0"


def git_add_commit_push(model_version: str, branch: str):
    """Adds, commits, and pushes the new model to a specific branch using SSH."""

    # Get repo URL from environment variable
    ssh_url = os.getenv("GIT_SSH_URL")

    if not ssh_url:
        raise ValueError("GIT_SSH_URL environment variable is not set")

    repo_dir = "./"

    # Intialize a new Git repository at the root of the container
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        subprocess.run(["git", "init"], cwd=repo_dir, check=True)

    # Add the remote repository URL
    subprocess.run(["git", "remote", "add", "origin", ssh_url], cwd=repo_dir, check=True)

    # Git add
    subprocess.run(["git", "add", "data/models/"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "data/images/"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "data/names.json"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "data/index_to_label.json"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "data/model_version.txt"], cwd=repo_dir, check=True)

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
