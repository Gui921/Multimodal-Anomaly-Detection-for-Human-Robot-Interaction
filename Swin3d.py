import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

class Swin3D(nn.Module):
    def __init__(self, n_classes):
        super(Swin3D, self).__init__()

        self.swin = swin3d_t(weights = Swin3D_T_Weights.KINETICS400_V1)
        self.swin.head = nn.Identity()

        for param in self.swin.parameters():
            param.requires_grad = False  # Freeze backbone

        self.classifier = nn.Linear(768, n_classes)

    def forward(self, x):

        x = self.swin(x)

        x = self.classifier(x)

        return x
    
class Swin3D_fine_tune(nn.Module):
    def __init__(self, n_classes):
        super(Swin3D_fine_tune, self).__init__()

        self.swin = swin3d_t(weights = Swin3D_T_Weights.KINETICS400_V1)
        self.swin.head = nn.Identity()

        self.classifier = nn.Linear(768, n_classes)

    def forward(self, x):

        x = self.swin(x)

        x = self.classifier(x)

        return x