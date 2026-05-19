import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18BinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNet18BinaryClassifier, self).__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=True)
        # Modify the final layer for binary classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # Binary output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)  # Output probabilities
        return x