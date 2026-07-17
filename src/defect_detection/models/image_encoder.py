
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from defect_detection.utils import load_yaml_config


def build_image_encoder(embedding_dim: int | None = None,
                         unfreeze_from: str | None = None) -> nn.Module:
    """Build a ResNet18-based image embedding encoder.

    Loads ImageNet-pretrained ResNet18, freezes all layers before unfreeze_from, and
    replaces the final classification layer with a linear projection to embedding_dim.

    Args:
        embedding_dim: Size of the output embedding. Defaults to
            config/model_config.yaml's image_encoder.embedding_dim if not given.
        unfreeze_from: Name of the first child module to leave trainable; all
            preceding modules are frozen. Defaults to
            config/model_config.yaml's image_encoder.unfreeze_from if not given.

    Returns:
        A ResNet18 model mapping (batch, 3, 224, 224) -> (batch, embedding_dim).
    """
    if embedding_dim is None or unfreeze_from is None:
        config = load_yaml_config("config/model_config.yaml")["image_encoder"]
        embedding_dim = embedding_dim if embedding_dim is not None else config["embedding_dim"]
        unfreeze_from = unfreeze_from if unfreeze_from is not None else config["unfreeze_from"]

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    unfreeze = False
    for name, child in model.named_children():
        if name == unfreeze_from:
            unfreeze = True
        if unfreeze:
            for param in child.parameters():
                param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, embedding_dim)  # in_features = 512 for ResNet18

    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total