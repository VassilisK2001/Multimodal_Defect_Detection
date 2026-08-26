import numpy as np


def build_inspection_result(defect_proba: float, fault_proba: np.ndarray, threshold: float,
                             class_names: list[str]) -> dict:
    """Build the API's response payload from raw model outputs.

    Args:
        defect_proba: Sigmoid probability from the defect gate.
        fault_proba: (3,) softmax probabilities from the fault-type head.
        threshold: The tuned defect-gate decision threshold.
        class_names: Fault class names, in the same order as fault_proba.

    Returns:
        Dict with 'status' ('healthy'/'defective'), 'defect_probability',
        and, only if defective, 'fault_type'/'fault_confidence'.
    """
    if defect_proba < threshold:
        return {
            "status": "healthy", "defect_probability": defect_proba,
            "fault_type": None, "fault_confidence": None,
        }

    fault_idx = int(np.argmax(fault_proba))
    return {
        "status": "defective", "defect_probability": defect_proba,
        "fault_type": class_names[fault_idx], "fault_confidence": float(fault_proba[fault_idx]),
    }