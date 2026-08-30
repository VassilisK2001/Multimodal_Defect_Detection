"""
Shared test data builders and constants.
"""



import torch
from torch.utils.data import DataLoader, TensorDataset

CLASS_NAMES = ["outer_race", "inner_race", "ball"]

def make_synthetic_loader(n_samples: int, n_defective: int, batch_size: int = 8) -> DataLoader:
    """Build a synthetic DataLoader yielding (image, vib_features, is_defect,
    fault_class_idx, area_ratio) batches, matching MultimodalDefectDataset's output.
    """
    images = torch.randn(n_samples, 3, 224, 224)
    vib_features = torch.randn(n_samples, 5)

    is_defect = torch.zeros(n_samples)
    is_defect[:n_defective] = 1.0

    fault_class_idx = torch.full((n_samples,), -1, dtype=torch.long)
    if n_defective > 0:
        fault_class_idx[:n_defective] = torch.randint(0, 3, (n_defective,))

    area_ratio = torch.zeros(n_samples)

    dataset = TensorDataset(images, vib_features, is_defect, fault_class_idx, area_ratio)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)