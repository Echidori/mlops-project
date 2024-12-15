# Suppress macOS warning
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import json
import os
import logging
from config import CAMERA, FACE_DETECTION, PATHS, CONFIDENCE_THRESHOLD

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Initialize the camera with error handling

    Parameters:
        camera_index (int): Camera device index
    Returns:
        cv2.VideoCapture: Initialized camera object
    """
    try:
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            logger.error("Could not open webcam")
            return None

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return None


def load_names(filename: str) -> dict:
    """
    Load name mappings from JSON file

    Parameters:
        filename (str): Path to the JSON file containing name mappings
    Returns:
        dict: Dictionary mapping IDs to names
    """
    try:
        names_json = {}
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                content = fs.read().strip()
                if content:
                    names_json = json.loads(content)
        return names_json
    except Exception as e:
        logger.error(f"Error loading names: {e}")
        return {}


def save_name(face_id: int, face_name: str, filename: str) -> None:
    """
    Save name-ID mapping to JSON file

    Parameters:
        face_id (int): The identifier of user
        face_name (str): The user name
        filename (str): Path to the JSON file
    """
    try:
        names_json: Dict[str, str] = {}
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as fs:
                    content = fs.read().strip()
                    if content:  # Only try to load if file is not empty
                        names_json = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {filename}, starting fresh")
                names_json = {}

        names_json[str(face_id)] = face_name

        with open(filename, 'w') as fs:
            json.dump(names_json, fs, indent=4, ensure_ascii=False)
        logger.info(f"Saved name mapping for ID {face_id}")
    except Exception as e:
        logger.error(f"Error saving name mapping: {e}")
        raise


def get_face_id(directory: str) -> int:
    """
    Get the first available face ID by checking existing files.

    Parameters:
        directory (str): The path of the directory of images.
    Returns:
        int: The next available face ID
    """
    try:
        if not os.path.exists(directory):
            return 1

        user_ids = []
        for filename in os.listdir(directory):
            if filename.startswith('Users-'):
                try:
                    number = int(filename.split('-')[1])
                    user_ids.append(number)
                except (IndexError, ValueError):
                    continue

        return max(user_ids + [0]) + 1
    except Exception as e:
        logger.error(f"Error getting face ID: {e}")
        raise

def create_directory(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    except OSError as e:
        logger.error(f"Error creating directory {directory}: {e}")
        raise



if __name__ == "__main__":
    try:
        logger.info("Starting face recognition system...")

        # Initialize face recognizer
        # recognizer = cv2.face.LBPHFaceRecognizer_create()

        #model = torch.load("cnn_model.pth")
        #model.eval()

        if not os.path.exists(PATHS['trainer_file']):
            raise ValueError("Trainer file not found. Please train the model first.")
        # recognizer.read(PATHS['trainer_file'])

        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            raise ValueError("Error loading cascade classifier")

        # Initialize camera
        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise ValueError("Failed to initialize camera")

        # Load names
        names = load_names(PATHS['names_file'])
        if not names:
            logger.warning("No names loaded, recognition will be limited")

        logger.info("Face recognition started. Press 'ESC' to exit.")

        while True:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Recognize the face
                # id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (50, 50))

                #id = model.predict(face_img.flatten())

                #label = model.convert_prediction_to_label(id, names)

                # Check confidence and display result
                name = "label[0]"
                confidence_text = "N/A"

                # Display name and confidence
                cv2.putText(img, name, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence_text, (x + 5, y + h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

            cv2.imshow('Face Recognition', img)

            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        logger.info("Face recognition stopped")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()