
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def compute_defect_gate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute precision/recall/F1/support for the binary defect gate, per class.

    Args:
        y_true: (N,) binary array, 1 if defective else 0.
        y_pred: (N,) binary array, thresholded predictions.

    Returns:
        Dict with 'normal' and 'defect' keys, each holding precision/recall/f1/support.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0,
    )
    return {
        "normal": {
            "precision": precision[0], "recall": recall[0],
            "f1": f1[0], "support": int(support[0]),
        },
        "defect": {
            "precision": precision[1], "recall": recall[1],
            "f1": f1[1], "support": int(support[1]),
        },
    }


def compute_fault_type_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                class_names: list[str]) -> dict:
    """Compute precision/recall/F1/support for the fault-type head, per class,
    plus macro-averaged F1.

    Args:
        y_true: (N,) integer class-index array, defective samples only.
        y_pred: (N,) integer class-index array, defective samples only.
        class_names: Class names in index order.

    Returns:
        Dict with 'per_class' (one entry per class name, each with
        precision/recall/f1/support) and 'macro_f1'.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0,
    )
    per_class = {
        class_names[i]: {
            "precision": precision[i], "recall": recall[i],
            "f1": f1[i], "support": int(support[i]),
        }
        for i in range(len(class_names))
    }
    return {"per_class": per_class, "macro_f1": float(f1.mean())}

def compute_metrics_from_predictions(predictions: dict, class_names: list[str]) -> dict:
    """Compute defect-gate and fault-type metrics from a predictions dict.
 
    Args:
        predictions: A dict with returned predictions.
        class_names: Fault class names, in index order.
 
    Returns:
        Dict with "defect_metrics" and "fault_metrics".
    """
    return {
        "defect_metrics": compute_defect_gate_metrics(
            predictions["is_defect_true"], predictions["is_defect_pred"],
        ),
        "fault_metrics": compute_fault_type_metrics(
            predictions["fault_class_true"], predictions["fault_class_pred"], class_names,
        ),
    }