import torch

# CNN

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=500):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, X_train, y_train, num_epochs=5, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            inputs = torch.from_numpy(X_train).float()
            labels = torch.from_numpy(y_train).long()

            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    def predict(self, X_test):
        inputs = torch.from_numpy(X_test).float()
        outputs = self(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def convert_prediction_to_label(self, prediction, label_map):
        return [label_map[p] for p in prediction]