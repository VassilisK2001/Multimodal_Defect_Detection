import numpy as np

def find_defect_gate_examples(predictions: dict, n: int = 3) -> dict:
    """Find up to n example row indices each for correct-defective,
    false-negative, and false-positive defect-gate predictions.

    Args:
        predictions: Output of collect_test_predictions(), for the full test set,
            in the same row order as the test dataset.
        n: Maximum number of examples to return per case.

    Returns:
        Dict with keys "correct_defective", "false_negative", "false_positive",
        each mapping to a list of row indices.
    """
    y_true = predictions["is_defect_true"]
    y_pred = predictions["is_defect_pred"]

    def _first_n(mask: np.ndarray) -> list[int]:
        indices = np.where(mask)[0]
        return [int(i) for i in indices[:n]]

    return {
        "correct_defective": _first_n((y_true == 1) & (y_pred == 1)),
        "false_negative": _first_n((y_true == 1) & (y_pred == 0)),
        "false_positive": _first_n((y_true == 0) & (y_pred == 1)),
    }


def find_correct_normal_examples(predictions: dict) -> list[int]:
    """Find all row indices where the defect gate correctly predicted normal.
 
    Args:
        predictions: Output of collect_test_predictions(), for the full test
            set, in the same row order as the test dataset.
 
    Returns:
        A list of all matching row indices.
    """
    y_true = predictions["is_defect_true"]
    y_pred = predictions["is_defect_pred"]
    indices = np.where((y_true == 0) & (y_pred == 0))[0]
    return [int(i) for i in indices]


def find_fault_type_examples(predictions: dict, class_names: list[str], n: int = 3) -> dict:
    """Find up to n correct and n misclassified examples per fault class.

    Args:
        predictions: Output of collect_test_predictions().
        class_names: Fault class names, in index order.
        n: Maximum number of examples to return per case.

    Returns:
        Dict mapping each class name to {"correct": [entry, ...],
        "misclassified": [entry, ...]}, each a list of 
        {"row_index": int, "predicted_class": int}.
    """
    is_defect_true = predictions["is_defect_true"]
    defective_positions = np.where(is_defect_true == 1)[0]

    fault_true = predictions["fault_class_true"]
    fault_pred = predictions["fault_class_pred"]

    result = {}
    for class_idx, class_name in enumerate(class_names):
        class_mask = fault_true == class_idx
        correct_mask = class_mask & (fault_pred == class_idx)
        misclassified_mask = class_mask & (fault_pred != class_idx)

        result[class_name] = {
            "correct": _build_entries(correct_mask, defective_positions, fault_pred, n),
            "misclassified": _build_entries(misclassified_mask, defective_positions, fault_pred, n),
        }

    return result


def _build_entries(mask: np.ndarray, defective_positions: np.ndarray,
                    fault_pred: np.ndarray, n: int) -> list[dict]:
    """Build up to n {"row_index", "predicted_class"} entries for rows matching mask."""
    local_indices = np.where(mask)[0][:n]
    return [
        {"row_index": int(defective_positions[i]), "predicted_class": int(fault_pred[i])}
        for i in local_indices
    ]

def find_vibration_fails_fusion_succeeds_examples(vib_defect_pred: np.ndarray, fusion_defect_pred: np.ndarray,
                                                    is_defect_true: np.ndarray, n: int = 3) -> list:
    """Find up to n test rows where vibration-only's defect-gate prediction was
    wrong but the fusion model's was correct.
 
    Args:
        vib_defect_pred: (N,) vibration-only's thresholded defect predictions.
        fusion_defect_pred: (N,) fusion model's thresholded defect predictions.
        is_defect_true: (N,) ground truth, same row order.
        n: Maximum number of examples to return.
 
    Returns:
        A list of up to n row indices.
    """
    vib_wrong = vib_defect_pred != is_defect_true
    fusion_right = fusion_defect_pred == is_defect_true
    indices = np.where(vib_wrong & fusion_right)[0]
    return [int(i) for i in indices[:n]]
 