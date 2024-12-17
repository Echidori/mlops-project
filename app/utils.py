import cv2
import os
from config import PATHS

import requests

def refresh_model(model):
    """Load the model for face recognition."""
    try:
        # Initialize the face recognizer
        model = cv2.face.LBPHFaceRecognizer_create()

        if not os.path.exists(PATHS['trainer_file']):
            raise ValueError("Trainer file not found. Please train the model first.")

        model.read(PATHS['trainer_file'])
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    return model




def send_photos_to_server(person_name, files):
    """Send the photos to the server."""
    # url = "http://your-server-url/add_person"  # Remplacer par l'URL de votre serveur

    data = {"name": person_name}

    try:
        print("Sending photos to server...")

        # Affichage des noms des fichiers avant l'envoi
        for file in files:
            photo_name = file[1][0]  # Le nom du fichier est le premier élément du tuple
            print(f"Sending photo: {photo_name}")

        print("Photos sent successfully.")
        # Enlever les commentaires pour réellement envoyer les photos
        # response = requests.post(url, data=data, files=files)
        # response.raise_for_status()

        # return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error sending photos: {e}")
        return None