import json
import os
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import RedirectResponse, FileResponse
from dotenv import load_dotenv
from typing import List
import shutil

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

def git_add_commit_push(model_version: str, branch: str):
    """Clones the repository, copies necessary files, adds, commits, and pushes the new model to a specific branch using SSH."""
    ssh_url = os.getenv("GIT_SSH_URL")

    if not ssh_url:
        raise ValueError("GIT_SSH_URL environment variable is not set")

    # Directory for cloning the repository
    repo_dir = "./temp_repo"

    print("Configuring Git user...")
    subprocess.run(["git", "config", "--global", "user.name", "JulienSchaff"], check=True)
    subprocess.run(["git", "config", "--global", "user.email", "julien.schaffauser@epita.fr"], check=True)

    # Clone the repository with the given branch
    print(f"Cloning repository and checking out branch '{branch}'...")
    subprocess.run(["git", "clone", "-b", branch, ssh_url, repo_dir], check=True)

    # Define the source directory (where the files are currently stored)
    source_dir = "./"  # Replace with the directory that contains the files to be added

    # List of files and directories to copy
    files_to_copy = [
        "data/models/",
        "data/images/",
        "data/names.json",
        "data/index_to_label.json",
        "data/model_version.txt"
    ]

    # Copy the necessary files into the cloned repo directory
    print("Copying necessary files to the cloned repository...")
    for item in files_to_copy:
        source_path = os.path.join(source_dir, item)
        dest_path = os.path.join(repo_dir, item)

        if os.path.isdir(source_path):
            # Copy the entire directory
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
        else:
            # Copy the single file
            shutil.copy2(source_path, dest_path)

    # Change to the cloned repository directory
    os.chdir(repo_dir)

    # Check the Git status to verify added files
    print("Checking Git status to verify staged changes...")
    result = subprocess.run(["git", "status"], check=True, capture_output=True, text=True)

    # Print the output of git status
    print("Git status output:")
    print(result.stdout)

    # Add the copied files to Git
    print("Adding new files to Git...")
    subprocess.run(["git", "add", "-f", "data/models/"], check=True)
    subprocess.run(["git", "add", "-f", "data/images/"], check=True)
    subprocess.run(["git", "add", "-f", "data/names.json"], check=True)
    subprocess.run(["git", "add", "-f", "data/index_to_label.json"], check=True)
    subprocess.run(["git", "add", "-f", "data/model_version.txt"], check=True)

    # Check the Git status to verify added files
    print("Checking Git status to verify staged changes...")
    result = subprocess.run(["git", "status"], check=True, capture_output=True, text=True)

    # Print the output of git status
    print("Git status output:")
    print(result.stdout)

    # Commit the changes
    commit_message = f"Add new model version {model_version}"
    print(f"Committing changes with message: '{commit_message}'")
    subprocess.run(["git", "commit", "-m", commit_message], check=True)

    # Push the changes
    print(f"Pushing changes to branch '{branch}'...")
    subprocess.run(["git", "push"], check=True)

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
    else:
        model_version = "0"

    # Update the names file
    names_file = Path("../data/names.json")
    if names_file.exists():
        with open(names_file, "r") as f:
            names = f.read()
    else:
        names = "{}"

    # the format of the names file is a JSON object
    # {
    #     "name1": "label1",
    #     "name2": "label2",
    #     ...
    # }

    # Parse the JSON object
    names_dict = json.loads(names)

    # Add the new name to the dictionary
    names_dict[name] = model_version

    # Convert the dictionary back to a JSON object
    names = json.dumps(names_dict)

    # Write the updated names to the file
    with open(names_file, "w") as f:
        f.write(names)

    # Update the index_to_label file
    index_to_label_file = Path("../data/index_to_label.json")

    if index_to_label_file.exists():
        with open(index_to_label_file, "r") as f:
            index_to_label = f.read()
    else:
        index_to_label = "{}"

    # the format of the index_to_label file is a JSON object
    # {
    #    "0": "1",
    #    "1": "2",
    #    ...
    # }

    # Parse the JSON object
    index_to_label_dict = json.loads(index_to_label)

    last_index = -1
    if index_to_label:
        last_index = max(int(k) for k in index_to_label_dict.keys())

    # Ajouter la nouvelle paire "last_index + 1": "last_index + 2"
    index_to_label_dict[str(last_index + 1)] = last_index + 2

    # Sauvegarder le fichier avec les données mises à jour
    with open(index_to_label_file, "w") as f:
        json.dump(index_to_label_dict, f, indent=4)

    # Update the model version file
    model_version_file = Path("../data/model_version.txt")
    with open(model_version_file, "w") as f:
        f.write(str(int(model_version) + 1))

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
