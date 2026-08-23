import numpy as np
from sklearn.metrics import precision_recall_curve


def select_recall_constrained_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                                         target_recall: float = 0.95) -> dict:
    """Select the lowest defect-gate threshold achieving at least target_recall,
    maximizing precision subject to that recall floor.

    Args:
        y_true: (N,) binary ground truth.
        y_proba: (N,) predicted defect probabilities.
        target_recall: Minimum acceptable recall.

    Returns:
        Dict with 'threshold', 'precision', 'recall' achieved at that
        threshold, and 'target_recall_achieved'. If no threshold reaches
        target_recall, falls back to the threshold giving maximum achievable
        recall, with 'target_recall_achieved': False.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]

    meets_target = recall >= target_recall
    if not meets_target.any():
        idx = int(np.argmax(recall))
        return {
            "threshold": float(thresholds[idx]), "precision": float(precision[idx]),
            "recall": float(recall[idx]), "target_recall_achieved": False,
        }

    # Among thresholds meeting the recall floor, the last True index gives the
    # highest threshold and best precision.
    idx = int(np.where(meets_target)[0][-1])
    return {
        "threshold": float(thresholds[idx]), "precision": float(precision[idx]),
        "recall": float(recall[idx]), "target_recall_achieved": True,
    }


def check_threshold_transfers_to_final_model(threshold: float, val_is_defect_true: np.ndarray,
                                              val_defect_proba: np.ndarray, target_recall: float = 0.95,
                                              tolerance: float = 0.05) -> dict:
    """Check a threshold's recall/precision on a separate set of predictions.

    Args:
        threshold: Decision threshold to check.
        val_is_defect_true: (N,) ground truth.
        val_defect_proba: (N,) predicted probabilities.
        target_recall: The recall floor the threshold was chosen to satisfy.
        tolerance: Maximum acceptable drop below target_recall before
            'diverges' is set to True.

    Returns:
        Dict with 'recall', 'precision' at this threshold, and 'diverges'
        (True if recall falls more than `tolerance` below target_recall).
    """
    val_pred = (val_defect_proba >= threshold).astype(int)
    tp = int(((val_pred == 1) & (val_is_defect_true == 1)).sum())
    fn = int(((val_pred == 0) & (val_is_defect_true == 1)).sum())
    fp = int(((val_pred == 1) & (val_is_defect_true == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return {
        "recall": recall, "precision": precision,
        "diverges": bool(recall < target_recall - tolerance),
    }