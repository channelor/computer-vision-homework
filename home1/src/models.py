import torch.nn as nn
from torchvision.models import resnet18


def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """Build a ResNet-18 variant suitable for small CIFAR-10 images."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_pretrained_resnet18(
    num_classes: int = 10,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Load an ImageNet-pretrained ResNet-18 backbone and replace the classifier."""
    from torchvision.models import ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
