import numpy as np
from sklearn.metrics import precision_recall_curve


def _select_threshold_for_recall(y_true: np.ndarray, y_proba: np.ndarray,
                                  target_recall: float) -> dict:
    """Select the highest threshold achieving at least target_recall on one
    sample, maximizing precision subject to that recall floor.

    Args:
        y_true: (N,) binary ground truth.
        y_proba: (N,) predicted probabilities.
        target_recall: Minimum acceptable recall.

    Returns:
        Dict with 'threshold', 'precision', 'recall', 'target_recall_achieved'.
        If no threshold reaches target_recall, falls back to the threshold
        giving maximum achievable recall.
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

    idx = int(np.where(meets_target)[0][-1])
    return {
        "threshold": float(thresholds[idx]), "precision": float(precision[idx]),
        "recall": float(recall[idx]), "target_recall_achieved": True,
    }


def bootstrap_threshold_distribution(y_true: np.ndarray, y_proba: np.ndarray,
                                      target_recall: float = 0.95, n_bootstrap: int = 1000,
                                      seed: int = 42) -> dict:
    """Estimate the defect-gate threshold's sampling distribution via
    stratified bootstrap resampling of (y_true, y_proba).

    Each resample draws the positive and negative rows separately, with
    replacement, preserving the original class balance.

    Args:
        y_true: (N,) binary ground truth.
        y_proba: (N,) predicted probabilities, same source as y_true.
        target_recall: Minimum acceptable recall.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with:
            'thresholds', 'precisions', 'recalls': (n_bootstrap,) arrays, one
                value per resample.
            'mean_threshold', 'std_threshold', 'mean_precision',
                'std_precision', 'mean_recall', 'std_recall': across all
                resamples.
            'n_target_achieved': count of resamples where target_recall was
                reached, out of n_bootstrap.
            'n_bootstrap': the input n_bootstrap, echoed back for convenience.
    """
    rng = np.random.default_rng(seed)
    positive_idx = np.where(y_true == 1)[0]
    negative_idx = np.where(y_true == 0)[0]

    thresholds, precisions, recalls = [], [], []
    n_target_achieved = 0

    for _ in range(n_bootstrap):
        resampled_pos = rng.choice(positive_idx, size=len(positive_idx), replace=True)
        resampled_neg = rng.choice(negative_idx, size=len(negative_idx), replace=True)
        resample_idx = np.concatenate([resampled_pos, resampled_neg])

        result = _select_threshold_for_recall(y_true[resample_idx], y_proba[resample_idx], target_recall)
        thresholds.append(result["threshold"])
        precisions.append(result["precision"])
        recalls.append(result["recall"])
        if result["target_recall_achieved"]:
            n_target_achieved += 1

    thresholds = np.array(thresholds)
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    return {
        "thresholds": thresholds, "precisions": precisions, "recalls": recalls,
        "mean_threshold": float(thresholds.mean()), "std_threshold": float(thresholds.std()),
        "mean_precision": float(precisions.mean()), "std_precision": float(precisions.std()),
        "mean_recall": float(recalls.mean()), "std_recall": float(recalls.std()),
        "n_target_achieved": n_target_achieved, "n_bootstrap": n_bootstrap,
    }