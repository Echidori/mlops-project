import cv2
import json
import os

import numpy as np
import torch
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout

# Configurations
from config import CAMERA, FACE_DETECTION, PATHS, TRAINING

from utils import refresh_model, send_photos_to_server, get_names, load_label_map

# Helper Functions
def initialize_camera(camera_index=0):
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise Exception("Camera not available")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA["width"])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA["height"])
    return cam

def save_name(face_id, face_name, filename):
    names = {}
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            content = file.read().strip()
            if content:
                names = json.loads(content)
    names[str(face_id)] = face_name
    with open(filename, 'w') as file:
        json.dump(names, file, indent=4)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class DefaultPage(Screen):
    def __init__(self, model=None, names=None, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.img_widget = Image()
        self.model = model  # Add model here
        self.names = names  # Add names dictionary
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.img_widget)
        self.add_widget(layout)
        self.label_map = {}

    def on_enter(self):
        index_to_label = load_label_map()
        for idx, face_id in index_to_label.items():
            name = self.names.get(str(face_id), "Unknown")
            self.label_map[face_id] = name

        self.capture = initialize_camera(CAMERA["index"])
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(PATHS["cascade_file"])
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=FACE_DETECTION["scale_factor"],
                    minNeighbors=FACE_DETECTION["min_neighbors"],
                    minSize=FACE_DETECTION["min_size"],
                )
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Recognize the face
                    if self.model:

                        face_img = gray[y:y + h, x:x + w]
                        face_img = cv2.resize(face_img, (50, 50))
                        face_img = face_img.astype(np.float32) / 255.0
                        face_img = face_img.reshape(1, 1, 50, 50)  # Reshape to match the input shape

                        input_tensor = torch.from_numpy(face_img)

                        with torch.no_grad():
                            outputs = self.model(input_tensor)

                            _, predicted = torch.max(outputs.data, 1)
                            predicted_idx = predicted.item()

                        # Get the name from the names dictionary based on the ID
                        name = self.label_map.get(predicted_idx, "Unknown")

                        # Display name and confidence
                        cv2.putText(frame, name, (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
                texture.blit_buffer(cv2.flip(frame, 0).tobytes(), colorfmt="bgr", bufferfmt="ubyte")
                self.img_widget.texture = texture

    def on_leave(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        Clock.unschedule(self.update)


class AddPersonPage(Screen):
    def __init__(self, names=None, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.face_cascade = cv2.CascadeClassifier(PATHS["cascade_file"])
        self.img_widget = Image()
        self.person_name = None
        self.names = names
        self.face_id = None
        self.count = 0
        self.is_capturing = False

        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.img_widget)
        self.add_widget(layout)

    def on_enter(self):
        self.capture = initialize_camera(CAMERA["index"])
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=FACE_DETECTION["scale_factor"],
                    minNeighbors=FACE_DETECTION["min_neighbors"],
                    minSize=FACE_DETECTION["min_size"],
                )
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Capture faces if we are in capturing mode and have not yet reached the sample limit
                    if self.is_capturing and self.count < TRAINING["samples_needed"]:
                        face_img = gray[y: y + h, x: x + w]
                        img_path = f'{PATHS["image_dir"]}/Users-{self.face_id}-{self.count + 1}.jpg'
                        cv2.imwrite(img_path, face_img)
                        self.count += 1

                    # Display photo count on the image
                    cv2.putText(frame, f"{self.count}/{TRAINING['samples_needed']}",
                                (x + w - 120, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Once we have captured enough samples, stop capturing and send the photos
                if self.count >= TRAINING["samples_needed"]:
                    self.is_capturing = False
                    print("Reached sample limit. Stopping capture and sending photos.")
                    # Call the function to send photos to the server
                    self.send_photos_to_server_and_complete()

                # Display the video feed on the screen
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
                texture.blit_buffer(cv2.flip(frame, 0).tobytes(), colorfmt="bgr", bufferfmt="ubyte")
                self.img_widget.texture = texture

    def start_capture(self):
        popup_layout = GridLayout(cols=1, padding=10)
        popup_label = Label(text="Enter your name:")
        name_input = TextInput(multiline=False)
        submit_btn = Button(text="Submit")

        def on_submit(instance):
            self.person_name = name_input.text.strip()
            if not self.person_name:
                return

            self.face_id = len(self.names.keys()) + 1
            create_directory(PATHS["image_dir"])
            save_name(self.face_id, self.person_name, PATHS["names_file"])
            self.is_capturing = True
            self.count = 0
            popup.dismiss()

        submit_btn.bind(on_press=on_submit)
        popup_layout.add_widget(popup_label)
        popup_layout.add_widget(name_input)
        popup_layout.add_widget(submit_btn)
        popup = Popup(title="Name Entry", content=popup_layout, size_hint=(0.8, 0.4))
        popup.open()

    def on_leave(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        Clock.unschedule(self.update)

    def send_photos_to_server_and_complete(self):
        # Créer le chemin vers le dossier des images de l'utilisateur
        photos_folder = f'{PATHS["image_dir"]}'

        # Lister tous les fichiers dans le dossier correspondant à face_id
        photo_files = [
            f for f in os.listdir(photos_folder)
            if f.startswith(f'Users-{self.face_id}-') and f.endswith('.jpg')
        ]

        # Vérifier si des photos sont trouvées
        if not photo_files:
            print("No photos found to send.")
            return

        # Préparer les fichiers pour l'envoi
        files = []
        for photo_filename in photo_files:
            photo_path = os.path.join(photos_folder, photo_filename)
            files.append(("photos", (photo_filename, open(photo_path, 'rb'), "image/jpeg")))

        # Lancer l'envoi des photos au serveur
        response = send_photos_to_server(self.person_name, files)
        if response:
            print("Photos sent successfully.")
        else:
            print("Failed to send photos.")

        # Réinitialiser le compteur après l'envoi
        self.count = 0

class CameraApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_page = None
        self.names = self.load_names()
        self.model = None
        self.model_version = None
        self.update_model()

    def load_names(self):
        if os.path.exists(PATHS["names_file"]):
            with open(PATHS["names_file"], 'r') as file:
                return json.load(file)
        return {}

    def update_model(self):
        """Load the model for face recognition."""
        try:
            self.model = refresh_model(self.model)
            if self.model is not None:
                self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def build(self):
        sm = ScreenManager()
        self.default_page = DefaultPage(model=self.model, names=self.names, name="default")
        self.add_person_page = AddPersonPage(name="add_person", names=self.names)

        sm.add_widget(self.default_page)
        sm.add_widget(self.add_person_page)
        self.current_page = self.default_page

        root_layout = BoxLayout(orientation="vertical")
        btn_layout = BoxLayout(size_hint=(1, 0.1))

        switch_page_btn = Button(text="Add New Person")
        action_btn = Button(text="Refresh Model")

        switch_page_btn.bind(on_press=lambda *args: self.switch_page(sm, switch_page_btn, action_btn))
        btn_layout.add_widget(switch_page_btn)

        action_btn.bind(on_press=lambda *args: self.perform_action(sm))
        btn_layout.add_widget(action_btn)

        root_layout.add_widget(sm)
        root_layout.add_widget(btn_layout)
        return root_layout

    def switch_page(self, sm, switch_btn, action_btn):
        if sm.current == "default":
            sm.current = "add_person"
            switch_btn.text = "Go Back to Default Page"
            action_btn.text = "Start"
        else:
            sm.current = "default"
            switch_btn.text = "Add New Person"
            action_btn.text = "Refresh Model"

    def perform_action(self, sm):
        if sm.current == "default":
            print("Model refreshed.")  # Placeholder action
            self.update_model()  # Refresh the model
        elif sm.current == "add_person":
            self.add_person_page.start_capture()

if __name__ == "__main__":
    # Update the names file
    get_names()
    CameraApp().run()
