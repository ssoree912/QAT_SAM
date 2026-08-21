import torch.nn as nn
import torchvision.models as tvmodels


def resnet18_cifar(num_classes=100, pretrained_imagenet=True):
    """CIFAR-adapted torchvision ResNet-18:
        - conv1: 3x3 stride 1 (instead of 7x7 stride 2)
        - maxpool: Identity (32x32 input is too small for it)
        - fc:     num_classes
    Optionally inherits layer1~layer4 weights from ImageNet pretrained model.
    """
    weights = "IMAGENET1K_V1" if pretrained_imagenet else None
    m = tvmodels.resnet18(weights=weights)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def resnet50_cifar(num_classes=100, pretrained_imagenet=True):
    """CIFAR-adapted torchvision ResNet-50 (same stem changes as resnet18_cifar)."""
    weights = "IMAGENET1K_V1" if pretrained_imagenet else None
    m = tvmodels.resnet50(weights=weights)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
