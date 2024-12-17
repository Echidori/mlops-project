import torch.onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from cnn import Net
from config import PATHS

def load_images_and_labels():
    images = []
    labels = []

    # Get the directory of the current script
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the images directory
    images_dir = PATHS["image_dir"]


    if not os.path.exists(images_dir):
        print(f"Images directory {images_dir} does not exist.")
        return images, labels

    # Load the images and labels
    for file in os.listdir(images_dir):
        if file.endswith(".jpg"):
            file_parts = file.split("-")
            if len(file_parts) != 3:
                print(f"Skipping file with unexpected format: {file}")
                continue
            prefix, user_id_str, sample_id_ext = file_parts
            sample_id_str, ext = os.path.splitext(sample_id_ext)
            try:
                user_id = int(user_id_str)
                sample_id = int(sample_id_str)
            except ValueError:
                print(f"Skipping file with invalid user_id or sample_id: {file}")
                continue

            image_path = os.path.join(images_dir, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Failed to read image: {image_path}")
                continue
            image = cv2.resize(image, (50, 50))
            images.append(image)
            labels.append(user_id)


    labels = np.array(labels)
    return images, labels

def save_onnx_model(model, version):
    dummy_input = torch.randn(1, 1, 50, 50)  # Adjust the input size as needed
    torch.onnx.export(model, dummy_input, f"../data/models_full/model{version}.onnx")

    # Quantize the model
    quantized_model = quantize_dynamic(model_input=f"../data/models_full/model{version}.onnx", model_output= f"../data/models/model{version}.onnx", weight_type=QuantType.QUInt8)

    print('ONNX full precision model size (MB):', os.path.getsize(f"../data/models_full/model{version}.onnx")/(1024*1024))
    print('ONNX quantized model size (MB):', os.path.getsize(f"../data/models/model{version}.onnx")/(1024*1024))

if __name__ == "__main__":
    images, labels = load_images_and_labels()

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    if num_classes <= 0:
        print("Error: No classes found. Ensure that images and labels are loaded correctly.")
        exit()

    input_size = 50 * 50
    hidden_size = 500
    model = Net(input_size, num_classes, hidden_size)

    X_train = [image / 255.0 for image in images]  # Normalize images
    X_train = np.array(X_train)
    y_train = labels

    # Convert numpy data types to native Python data types
    label_to_index = {int(label): int(idx) for idx, label in enumerate(unique_labels)}
    index_to_label = {int(idx): int(label) for label, idx in label_to_index.items()}

    y_train_indices = np.array([label_to_index[int(label)] for label in y_train])

    with open("../data/index_to_label.json", "w") as f:
        json.dump(index_to_label, f)

    model.train_model(X_train, y_train_indices, num_epochs=50, learning_rate=0.001)

    torch.save(model, "cnn_model.pth")

    # The version of the model is the size of the "../data/models" directory
    model_version = len(os.listdir("../data/models")) + 1

    save_onnx_model(model, model_version)

    # Set the model version inside the "../data/model_version.txt" file
    with open("../data/model_version.txt", "w") as f:
        f.write(str(model_version))