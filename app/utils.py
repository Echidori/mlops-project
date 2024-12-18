import json
import os

import torch

from config import PATHS

import requests

from dotenv import load_dotenv

load_dotenv()



def refresh_model(model):
    """Load the model for face recognition."""
    try:
        # Initialize the face recognizer
        # call server to get model version
        url = os.getenv("SERVER_URL")
        if not url:
            raise ValueError("SERVER_URL environment variable is not set")

        response = requests.get(url + "/model_version")
        response.raise_for_status()
        model_version = response.json()["version"]
        print(f"Model version: {model_version}")

        current_model_version = "0"
        # Check if the model version is the same as the current model
        # The current model version is stored in the "model_version.txt" file
        if os.path.exists(PATHS['model_version_file']):
            with open(PATHS['model_version_file'], "r") as f:
                current_model_version = f.read().strip()
        if current_model_version == model_version and model is not None:
            print("Model is up to date.")
            return model

        print("Model is outdated. Loading new model...")

        response = requests.get(url + "/model")
        response.raise_for_status()

        print("Model loaded successfully.")

        model_file = response.content

        print(f"Model file size: {len(model_file)} bytes")

        with open(PATHS['model_file'], "wb") as f:
            f.write(model_file)

        model = torch.jit.load(PATHS['model_file'])

        print("Model loaded successfully.")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    return model

def load_label_map() -> dict:
    """Load the label map from the index_to_label.json file."""
    # Get the label map from the server

    url = os.getenv("SERVER_URL")
    if not url:
        raise ValueError("SERVER_URL environment variable is not set")

    try:
        response = requests.get(url + "/get_label_map")
        response.raise_for_status()
        index_to_label = response.json()["index_to_label"]

        index_to_label = json.loads(index_to_label)

        # Convert keys back to integers
        index_to_label = {int(k): int(v) for k, v in index_to_label.items()}
        return index_to_label
    except Exception as e:
        print(f"Error loading label map: {e}")
        return {}

def get_names():
    """Get the list of names from the server (as a JSON string) and put it in the names.json file."""
    try:
        url = os.getenv("SERVER_URL")
        if not url:
            raise ValueError("SERVER_URL environment variable is not set")

        response = requests.get(url + "/get_names")
        response.raise_for_status()
        names = response.json()["names"]

        with open(PATHS['names_file'], "w") as f:
            f.write(names)
        print("Names file updated successfully.")
    except Exception as e:
        print(f"Error updating names file: {e}")

def send_photos_to_server(person_name, files):
    """Send the photos to the server."""
    # Get the server URL from the environment variable
    url = os.getenv("SERVER_URL")
    if not url:
        raise ValueError("SERVER_URL environment variable is not set")

    data = {"name": person_name}

    try:
        print("Sending photos to server...")

        # Affichage des noms des fichiers avant l'envoi
        for file in files:
            photo_name = file[1][0]  # Le nom du fichier est le premier élément du tuple
            print(f"Sending photo: {photo_name}")

        print("Photos sent successfully.")
        response = requests.post(url + "/add_person", data=data, files=files)
        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error sending photos: {e}")
        return None