import pandas as pd
import torch
from PIL import Image

from defect_detection.data.augmentations import build_image_transform


def load_images_for_df(df: pd.DataFrame, project_root) -> torch.Tensor:
    """Load and transform each row's real image, matching
    MultimodalDefectDataset's own image pipeline.

    Args:
        df: Manifest rows, with an 'image_path' column.
        project_root: Project root, as returned by find_project_root().

    Returns:
        (len(df), 3, H, W) stacked, transformed image tensor.
    """
    transform = build_image_transform(training=False)
    images = [transform(Image.open(project_root / p).convert("RGB")) for p in df["image_path"]]
    return torch.stack(images)