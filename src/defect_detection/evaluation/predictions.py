
import numpy as np
import torch
from torch.utils.data import DataLoader

from defect_detection.models.batch_utils import forward_batch
from defect_detection.models.fusion_model import MultimodalDefectClassifier


@torch.no_grad()
def collect_test_predictions(model: MultimodalDefectClassifier, loader: DataLoader,
                              device: torch.device, defect_threshold: float = 0.5) -> dict:
    """Run a model over a DataLoader in eval mode and collect true/predicted arrays.

    Args:
        model: A trained MultimodalDefectClassifier.
        loader: DataLoader yielding (image, vib_features, is_defect, fault_class_idx,
            area_ratio) batches.
        device: Device to run inference on.
        defect_threshold: Probability threshold for the binary defect gate.

    Returns:
        Dict with:
            is_defect_true: (N,) int array.
            is_defect_pred: (N,) int array, thresholded at defect_threshold.
            defect_proba: (N,) float array, raw sigmoid probabilities — kept
                separately from is_defect_pred so the same collected data can be
                reused for threshold tuning without re-running inference.
            fault_class_true: (M,) int array, defective samples only.
            fault_class_pred: (M,) int array, argmax predictions, defective
                samples only.
    """
    model.eval()

    is_defect_true, is_defect_pred, defect_proba = [], [], []
    fault_class_true, fault_class_pred = [], []

    for batch in loader:
        defect_logit, fault_type_logits, is_defect, fault_class_idx = forward_batch(
            model, batch, device,
        )

        proba = torch.sigmoid(defect_logit.squeeze(1)).cpu().numpy()
        preds = (proba >= defect_threshold).astype(int)

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