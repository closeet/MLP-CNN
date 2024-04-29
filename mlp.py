import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Input layer
        self.fc1 = nn.Linear(32 * 32 * 3, 512)  # First fully connected layer
        # Hidden layers
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer
        self.fc3 = nn.Linear(256, 128)  # Third fully connected layer
        # Output layer
        self.fc4 = nn.Linear(128, 10)  # Output layer with 10 classes

    def forward(self, x):
        # Flatten the image from (3, 32, 32) to (3072)
        x = x.view(-1, 32*32*3)
        # Activation functions between layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Create the model
def mlp():
    return MLP()