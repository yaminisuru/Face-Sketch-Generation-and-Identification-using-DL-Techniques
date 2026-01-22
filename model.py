import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        self.fc = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # IMPORTANT
        return x
    