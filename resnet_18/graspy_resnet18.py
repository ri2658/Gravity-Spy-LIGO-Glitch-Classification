import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GraSpyResNet18(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()

        self.resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Modify first conv layer for 1-channel input
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )

        # Initialize weights by averaging RGB channels
        with torch.no_grad():
            self.resnet.conv1.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True)
            )

        # Replace classifier with correct number of classes
        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.resnet(x)