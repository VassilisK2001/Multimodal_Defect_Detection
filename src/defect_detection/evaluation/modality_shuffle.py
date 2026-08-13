
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.evaluation.metrics import compute_metrics_from_predictions
from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.models.batch_utils import forward_batch
from defect_detection.models.fusion_model import MultimodalDefectClassifier


def corrupt_batch(batch: tuple, corrupt_modality: str, method: str = "shuffle",
                   seed: int | None = None) -> tuple:
    """Corrupt one modality's tensor in a batch.

    Args:
        batch: A (image, vib_features, is_defect, fault_class_idx, area_ratio) tuple.
        corrupt_modality: "image" or "vibration" which tensor to corrupt.
        method: "shuffle" (permute across the batch dimension, mismatching pairs)
            or "zero" (replace with zeros).
        seed: Random seed for "shuffle", for reproducibility.

    Returns:
        A new batch tuple with the specified tensor corrupted; other elements unchanged.
    """
    images, vib_features, is_defect, fault_class_idx, area_ratio = batch

    if corrupt_modality == "image":
        images = _corrupt_tensor(images, method, seed=seed)
    elif corrupt_modality == "vibration":
        vib_features = _corrupt_tensor(vib_features, method, seed=seed)
    else:
        raise ValueError(f"Unknown corrupt_modality: {corrupt_modality}")

    return images, vib_features, is_defect, fault_class_idx, area_ratio


def _corrupt_tensor(tensor: torch.Tensor, method: str, seed: int | None = None) -> torch.Tensor:
    """Apply a corruption method to a single tensor.

    Args:
        tensor: The tensor to corrupt.
        method: "shuffle" (permute across dim 0) or "zero" (replace with zeros).
        seed: Random seed for "shuffle", for reproducibility. Ignored by "zero".

    Returns:
        The corrupted tensor.
    """
    if method == "shuffle":
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        perm = torch.randperm(tensor.size(0), generator=generator)
        return tensor[perm]
    if method == "zero":
        return torch.zeros_like(tensor)
    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def collect_predictions_with_corruption(model: MultimodalDefectClassifier, loader: DataLoader,
                                         device: torch.device, corrupt_modality: str,
                                         method: str = "shuffle", seed: int | None = None) -> dict:
    """Run a model over a DataLoader with one modality corrupted in every batch.

    Args:
        model: A trained MultimodalDefectClassifier.
        loader: DataLoader yielding (image, vib_features, is_defect, fault_class_idx,
            area_ratio) batches.
        device: Device to run inference on.
        corrupt_modality: "image" or "vibration" which input to corrupt.
        method: "shuffle" or "zero".
        seed: Base random seed for "shuffle", for reproducibility. Each batch uses
            seed + batch_index, so batches get distinct but reproducible permutations.

    Returns:
        Dict with is_defect_true, is_defect_pred, defect_proba, fault_class_true,
        fault_class_pred.
    """
    model.eval()
    is_defect_true, is_defect_pred, defect_proba = [], [], []
    fault_class_true, fault_class_pred = [], []

    for batch_idx, batch in enumerate(loader):
        batch_seed = seed + batch_idx if seed is not None else None
        batch = corrupt_batch(batch, corrupt_modality, method, seed=batch_seed)
        defect_logit, fault_type_logits, is_defect, fault_class_idx = forward_batch(model, batch, device)

        proba = torch.sigmoid(defect_logit.squeeze(1)).cpu().numpy()
        preds = (proba >= 0.5).astype(int)

        is_defect_true.append(is_defect.cpu().numpy().astype(int))
        is_defect_pred.append(preds)
        defect_proba.append(proba)

        defect_mask = is_defect.bool()
        if defect_mask.sum() > 0:
            fault_preds = fault_type_logits[defect_mask].argmax(dim=1).cpu().numpy()
            fault_class_pred.append(fault_preds)
            fault_class_true.append(fault_class_idx[defect_mask].cpu().numpy())

    return {
        "is_defect_true": np.concatenate(is_defect_true),
        "is_defect_pred": np.concatenate(is_defect_pred),
        "defect_proba": np.concatenate(defect_proba),
        "fault_class_true": np.concatenate(fault_class_true) if fault_class_true else np.array([], dtype=int),
        "fault_class_pred": np.concatenate(fault_class_pred) if fault_class_pred else np.array([], dtype=int),
    }

def run_modality_shuffle_test(model: MultimodalDefectClassifier, test_df: pd.DataFrame,
                               window_size: int, fs: int, class_names: list[str],
                               vib_mean: np.ndarray, vib_std: np.ndarray, device: torch.device,
                               method: str = "shuffle",
                               corrupt_modalities: tuple[str, ...] = ("image", "vibration"),
                               seed: int | None = 42) -> dict:
    """Compare a model's metrics on clean vs. per-modality-corrupted test data.

    Args:
        model: A trained MultimodalDefectClassifier.
        test_df: Test split manifest.
        window_size: Vibration window size in samples.
        fs: Vibration sampling rate in Hz.
        class_names: Fault class names, in index order.
        vib_mean: Vibration feature mean, from the model's training run.
        vib_std: Vibration feature std, from the model's training run.
        device: Device to run inference on.
        method: Corruption method: "shuffle" or "zero".
        corrupt_modalities: Which modalities to corrupt and report. Defaults to
            both; pass a subset (e.g. ("image",)) to test only one.
        seed: Random seed for "shuffle", for reproducibility.

    Returns:
        Dict with "baseline" plus one "<modality>_corrupted" entry per requested
        modality, each holding {"defect_metrics": ..., "fault_metrics": ...}.
    """
    test_dataset = MultimodalDefectDataset(
        test_df, window_size=window_size, fs=fs, training=False,
        vib_mean=vib_mean, vib_std=vib_std,
    )
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    baseline_predictions = collect_test_predictions(model, loader, device)
    results = {"baseline": compute_metrics_from_predictions(baseline_predictions, class_names)}

    for corrupt_modality in corrupt_modalities:
        predictions = collect_predictions_with_corruption(
            model, loader, device, corrupt_modality, method, seed=seed,
        )
        results[f"{corrupt_modality}_corrupted"] = compute_metrics_from_predictions(predictions, class_names)

    return results