import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class SiameseResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        f1 = self.backbone(img1)
        f2 = self.backbone(img2)
        diff = torch.abs(f1 - f2)
        return self.fc(diff)
