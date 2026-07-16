
import numpy as np
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def jitter(window: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Add Gaussian noise scaled to the window's own amplitude, simulating sensor noise."""
    noise = np.random.normal(0, sigma * np.std(window), size=window.shape)
    return window + noise


def scale(window: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Multiply the whole window by a random factor, simulating global amplitude
    drift (sensor sensitivity/mounting variation) while preserving the signal's shape."""
    factor = np.random.normal(1.0, sigma)
    return window * factor


def build_image_transform(training: bool) -> transforms.Compose:
    """ImageNet-normalized transform, matching the pretrained ResNet18 backbone's expected
    input distribution. Augmentation (rotation, brightness jitter) only applied when training."""
    if training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])