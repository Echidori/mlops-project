# Suppress macOS warning
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import json
import logging
import torch
from config import CAMERA, FACE_DETECTION, PATHS
import numpy as np
import json


from cnn import Net

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
        with open(filename, 'r') as f:
            names = json.load(f)
        return names
    except Exception as e:
        logger.error(f"Error loading names: {e}")
        return {}
    
def load_label_map(filename: str) -> dict:
    try:
        with open(filename, 'r') as f:
            index_to_label = json.load(f)
        # Convert keys back to integers
        index_to_label = {int(k): int(v) for k, v in index_to_label.items()}
        return index_to_label
    except Exception as e:
        logger.error(f"Error loading label map: {e}")
        return {}

if __name__ == "__main__":
    try:
        logger.info("Starting face recognition system...")

        # Load the entire model
        model = torch.load("src/cnn_model.pth", weights_only=False)
        model.eval()

        logger.info("Model loaded successfully.")

        logger.info(f"{PATHS['cascade_file']}")

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

        index_to_label = load_label_map("data/index_to_label.json")

        # Create label map: index (from model output) -> name
        label_map = {}
        for idx, face_id in index_to_label.items():
            name = names.get(str(face_id), "Unknown")
            label_map[idx] = name


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
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Recognize the face
                #id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (50, 50))
                face_img = face_img.astype(np.float32) / 255.0
                face_img = face_img.reshape(1, 1, 50, 50)  # Reshape to match the input shape

                input_tensor = torch.from_numpy(face_img)

                # Make prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_idx = predicted.item()

                name = label_map.get(predicted_idx, "Unknown")
                
                # Display name and confidence
                cv2.putText(img, name, (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                """cv2.putText(img, confidence_text, (x+5, y+h-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)"""
            
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