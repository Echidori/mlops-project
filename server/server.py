import os
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import RedirectResponse, FileResponse
from dotenv import load_dotenv
from typing import List

server = FastAPI()

DATA_DIR = os.getenv("DATA_DIR", "../data/")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

load_dotenv()


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
    model_version_path = Path(DATA_DIR) / "model_version.txt"
    with open(model_version_path, "r") as f:
        model_version = f.read().strip()

    model_path = Path(DATA_DIR) / f"models/model-{model_version}.pth"

    print(f"Returning model version {model_version}")
    return FileResponse(model_path, media_type='application/octet-stream')


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
        print("Starting model training...")
        result = subprocess.run(
            ["python", "../src/cnn_train.py"],
            capture_output=True,
            text=True,
            check=True
        )
        print("Model training complete.")
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
    ssh_url = os.getenv("GIT_SSH_URL")

    if not ssh_url:
        raise ValueError("GIT_SSH_URL environment variable is not set")

    repo_dir = "./"

    print("Configuring Git user...")
    subprocess.run(["git", "config", "--global", "user.name", "JulienSchaff"], check=True)
    subprocess.run(["git", "config", "--global", "user.email", "julien.schaffauser@epita.fr"], check=True)

    # Initialize a new Git repository at the root of the container if necessary
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        print("Initializing Git repository...")
        subprocess.run(["git", "init"], cwd=repo_dir, check=True)

    # Add the remote repository URL if not already added
    remotes = subprocess.run(["git", "remote"], cwd=repo_dir, capture_output=True, text=True, check=True)
    if "origin" not in remotes.stdout:
        print("Adding remote 'origin'...")
        subprocess.run(["git", "remote", "add", "origin", ssh_url], cwd=repo_dir, check=True)

    # Fetch the remote branch if it exists
    print(f"Fetching branch '{branch}' from remote...")
    subprocess.run(["git", "fetch", "origin", branch], cwd=repo_dir, check=True)

    # Checkout the branch (force it)
    print(f"Checking out branch '{branch}'...")
    subprocess.run(["git", "checkout", "-f", f"origin/{branch}"], cwd=repo_dir, check=True)

    # Rebase on top of the remote branch
    print(f"Rebasing local changes on top of the remote '{branch}'...")
    try:
        subprocess.run(["git", "rebase", "origin/" + branch], cwd=repo_dir, check=True)
        print("Rebase successful.")
    except subprocess.CalledProcessError:
        print("Rebase failed. Resolving conflicts...")
        # If rebase fails, you can resolve conflicts manually here.
        # In case of a conflict, you would use the following:
        subprocess.run(["git", "rebase", "--abort"], cwd=repo_dir, check=True)  # Abort the current rebase

        # Optionally, you can implement conflict resolution here based on your workflow.
        raise Exception("Rebase failed and conflicts need to be resolved manually.")

    # Add all the files
    print("Adding new files to Git...")
    subprocess.run(["git", "add", "-f", "data/models/"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "-f", "data/images/"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "-f", "data/names.json"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "-f", "data/index_to_label.json"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "-f", "data/model_version.txt"], cwd=repo_dir, check=True)

    # Commit the changes
    commit_message = f"Add new model version {model_version}"
    print(f"Committing changes with message: '{commit_message}'")
    subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_dir, check=True)

    # Push the changes
    print(f"Pushing changes to branch '{branch}'...")
    subprocess.run(["git", "push", "origin", f"{branch}:{branch}"], cwd=repo_dir, check=True)
    print(f"Successfully pushed changes to branch '{branch}'.")


@server.post("/add_person")
async def add_person(
        name: str = Form(...),
        photos: List[UploadFile] = File(...)):
    # Save photos
    images_dir = Path(DATA_DIR) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for photo in photos:
        file_path = images_dir / photo.filename
        with open(file_path, "wb") as f:
            f.write(await photo.read())

    # Train the model
    print(f"Starting training for person '{name}'...")
    model_version = train_model()

    # Ensure that model training is complete before continuing
    if model_version == "0":
        return {"error": "Model training failed or not completed properly."}

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
