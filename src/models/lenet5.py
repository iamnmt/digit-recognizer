import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=84, out_features=10),
            nn.Softmax(dim=1)
        )
    def forward(self, X):
        return self.classifier(self.extractor(X))