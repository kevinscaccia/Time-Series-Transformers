import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, input_len, model_dim=64):
        super(SimpleCNN, self).__init__()
        self.conv1d = nn.Conv1d(input_len, model_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(model_dim, 50)
        self.fc2 = nn.Linear(50, 1)
        
    def forward(self,x):
        # cnn filter
        x = self.relu(self.conv1d(x))
        # bridge between cnn and dnn
        x = self.flatten(x)
        # dense 1
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x