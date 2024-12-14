import os

from cnn import Net
import cv2
import torch

def train_cnn(X_train, y_train, num_epochs=5, learning_rate=0.001):
    # Initialize the model
    model = Net(input_size=784, hidden_size=500, num_classes=10)

    # Train the model
    model.train_model(X_train, y_train, num_epochs=num_epochs, learning_rate=learning_rate)

    return model


def load_images_and_labels():
    images = []
    labels = []

    # Load the images and labels
    for file in os.listdir("images"):
        if file.endswith(".jpg"):
            user_id, sample_id = file.split("-")[1:]
            image = cv2.imread(f"images/{file}")

            image = cv2.resize(image, (50, 50))

            images.append(image)
            labels.append(user_id)

    # Convert the labels to integers
    label_map = {label: i for i, label in enumerate(set(labels))}
    labels = [label_map[label] for label in labels]

    return images, labels


if __name__ == "__main__":
    # Load the images and labels
    # from the images folder, we have the following files:
    # - Users-{user_id}-{sample_id}.jpg
    # from the names.json file, we have the following data:
    # {
    #    "${user_id}": "${user_name}"
    # }
    # We need to load the images and labels from the files
    # and convert the labels to integers
    images, labels = load_images_and_labels()

    # Train the CNN model
    input_size = images[0].shape[0] * images[0].shape[1]
    num_classes = len(set(labels))
    model = Net(input_size, num_classes)

    X_train = [image.flatten() for image in images]
    y_train = labels

    model.train_model(X_train, y_train, num_epochs=5, learning_rate=0.001)

    # Save the model
    model.save_model("cnn_model.pth")

    print("CNN model trained and saved.")