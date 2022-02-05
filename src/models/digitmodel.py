import torch
import torch.nn as nn

from .extractors import ResNetExtractor

class BaseModel(nn.Module):
    def __init__(self, num_classes=10, version=50, from_pretrained=True):
        super().__init__()
        self.extractor = ResNetExtractor(version)

        self.feature_dim = self.extractor.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x