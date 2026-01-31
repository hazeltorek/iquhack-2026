import argparse
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Simple Logistic Regression Model
class MulticlassLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
 
    def forward(self, x):
        out = self.linear(x)
        return out
 
def main(args):
    # print(args.tasks)
    # print(args.circuits)
    # print(args.id_map)
    # print(args.out)

    X = [] # Input Feature Data 
    y = [] # Threshold Values

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)


    threshold_classes = [1, 2, 4, 8, 16, 32, 64, 128, 256] # Output 
    feature_size = 10                                      # Feature length
    
    # Create a DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=feature_size, shuffle=True)

    
    model = MulticlassLogisticRegression(feature_size, threshold_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train Model
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Test Model
    new_X = torch.tensor(np.random.randn(5, 10), dtype=torch.float32)
    with torch.no_grad():
        outputs = model(new_X)
        _, predicted = torch.max(outputs, 1)
        #print('Predicted classes:', predicted)

    # Model Accuracy
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
        print('Accuracy:', accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type = str, default="data\hackathon_public.json")
    parser.add_argument("--circuits", type = str, default="circuits")
    parser.add_argument("--id-map", type = str, default="data\holdout_public.json")
    parser.add_argument("--out", type = str, default="predictions.json")
    args = parser.parse_args()

    main(args)
                   