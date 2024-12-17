import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=500):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, X_train, y_train, num_epochs=5, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if isinstance(X_train, list):
            X_train = np.array(X_train)
        if isinstance(y_train, list):
            y_train = np.array(y_train)

        for epoch in range(num_epochs):
            inputs = torch.from_numpy(X_train).float().unsqueeze(1)  # Add channel dimension
            labels = torch.from_numpy(y_train).long()

            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    def predict(self, X_test):
        inputs = torch.from_numpy(X_test).unsqueeze(1).float()  # Add channel dimension
        outputs = self(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted.numpy()


    def convert_prediction_to_label(self, prediction, label_map):
        return [label_map.get(str(p), "Unknown") for p in prediction]

    def save_model(self, path):
        torch.save(self.state_dict(), path)