import cv2
import json
import os
#import torch
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# Configurations (Replace with your actual paths and parameters)
CAMERA = {"index": 0, "width": 640, "height": 480}
FACE_DETECTION = {"scale_factor": 1.1, "min_neighbors": 5, "min_size": (30, 30)}
TRAINING = {"samples_needed": 10}
PATHS = {
    "cascade_file": "cascade.xml",
    "image_dir": "../src/images",
    "names_file": "../data/names.json",
}

# Helper Functions
def initialize_camera(camera_index=0):
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise Exception("Camera not available")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA["width"])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA["height"])
    return cam


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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


class DefaultPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None  # Initialize without the camera
        self.img_widget = Image()
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.img_widget)
        self.add_widget(layout)

    def on_enter(self):
        # Initialize the camera when the screen is entered
        self.capture = initialize_camera(CAMERA["index"])
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                # Process frame for face recognition (Dummy Example)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
                texture.blit_buffer(cv2.flip(frame, 0).tobytes(), colorfmt="bgr", bufferfmt="ubyte")
                self.img_widget.texture = texture

    def on_leave(self):
        # Release the camera when leaving the screen
        if self.capture:
            self.capture.release()
            self.capture = None
        Clock.unschedule(self.update)


class AddPersonPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None  # Initialize without the camera
        self.face_cascade = cv2.CascadeClassifier(PATHS["cascade_file"])
        self.img_widget = Image()
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.img_widget)
        self.add_widget(layout)
        self.count = 0
        self.face_id = None

    def on_enter(self):
        # Prompt for user name and initialize
        self.capture = initialize_camera(CAMERA["index"])
        face_name = "Enter a valid name here"  # Replace with actual input logic
        self.face_id = len(os.listdir(PATHS["image_dir"])) + 1
        create_directory(PATHS["image_dir"])
        save_name(self.face_id, face_name, PATHS["names_file"])
        Clock.schedule_interval(self.capture_faces, 1.0 / 30.0)

    def capture_faces(self, dt):
        if self.capture:
            ret, frame = self.capture.read()
            if not ret:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION["scale_factor"],
                minNeighbors=FACE_DETECTION["min_neighbors"],
                minSize=FACE_DETECTION["min_size"],
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if self.count < TRAINING["samples_needed"]:
                    face_img = gray[y : y + h, x : x + w]
                    img_path = f'./{PATHS["image_dir"]}/Users-{self.face_id}-{self.count + 1}.jpg'
                    cv2.imwrite(img_path, face_img)
                    self.count += 1

            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
            texture.blit_buffer(cv2.flip(frame, 0).tobytes(), colorfmt="bgr", bufferfmt="ubyte")
            self.img_widget.texture = texture

    def on_leave(self):
        # Release the camera when leaving the screen
        if self.capture:
            self.capture.release()
            self.capture = None
        Clock.unschedule(self.capture_faces)


class CameraApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(DefaultPage(name="default"))
        sm.add_widget(AddPersonPage(name="add_person"))
        root_layout = BoxLayout(orientation="vertical")
        btn_layout = BoxLayout(size_hint=(1, 0.1))

        default_btn = Button(text="Default Page")

        def switch_to_default(*args):
            sm.current = "default"

        default_btn.bind(on_press=switch_to_default)

        btn_layout.add_widget(default_btn)

        add_person_btn = Button(text="Add New Person")

        def switch_to_add_person(*args):
            sm.current = "add_person"

        add_person_btn.bind(on_press=switch_to_add_person)
        btn_layout.add_widget(add_person_btn)

        root_layout.add_widget(sm)
        root_layout.add_widget(btn_layout)
        return root_layout


if __name__ == "__main__":
    CameraApp().run()
