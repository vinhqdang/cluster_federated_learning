"""Pretrained ResNet-18 split into a frozen backbone and a trainable head.

All federated methods train the same head, ``layer4 + global pooling + fc`` of an
ImageNet-pretrained ResNet-18 (8.4M parameters). The frozen part (stem, layer1-3,
batch-norm statistics included) is applied once to every image, so that the head
is trained on cached layer-3 feature maps. This is exact, because the frozen part
has no trainable parameters and no augmentation is used.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resnet18(pretrained):
    import torchvision
    return torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)


class L4Head(nn.Module):
    """layer4 + average pooling + linear classifier on layer-3 feature maps."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.layer4 = _resnet18(pretrained).layer4
        self.fc = nn.Linear(512, num_classes)

    def features(self, x):
        return F.adaptive_avg_pool2d(self.layer4(x), 1).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():          # batch-norm statistics stay frozen (FedAvg
            if isinstance(m, nn.BatchNorm2d):   # cannot average running statistics)
                m.eval()
        return self


class FrozenBackbone(nn.Module):
    """Stem + layer1-3 of a pretrained ResNet-18 (always in eval mode)."""

    def __init__(self, pretrained=True):
        super().__init__()
        m = _resnet18(pretrained)
        self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x_uint8):
        x = (x_uint8.float() / 255.0 - self.mean) / self.std
        return self.body(x)

    def featurize(self, x_uint8, batch=256):
        dev = self.mean.device
        out = [self(x_uint8[s:s + batch].to(dev)) for s in range(0, len(x_uint8), batch)]
        return torch.cat(out) if out else torch.zeros(0)
