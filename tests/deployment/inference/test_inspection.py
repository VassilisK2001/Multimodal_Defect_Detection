import sys
from pathlib import Path

import numpy as np

DEPLOYMENT_DIR = Path(__file__).resolve().parents[3] / "deployment"
sys.path.insert(0, str(DEPLOYMENT_DIR))

from inference.inspection import build_inspection_result 


CLASS_NAMES = ["outer_race", "inner_race", "ball"]


def test_below_threshold_is_healthy_with_no_fault_fields():
    result = build_inspection_result(0.1, np.array([0.5, 0.3, 0.2]), threshold=0.3, class_names=CLASS_NAMES)

    assert result["status"] == "healthy"
    assert result["defect_probability"] == 0.1
    assert result["fault_type"] is None
    assert result["fault_confidence"] is None


def test_at_threshold_boundary_is_defective():
    result = build_inspection_result(0.3, np.array([0.5, 0.3, 0.2]), threshold=0.3, class_names=CLASS_NAMES)

    assert result["status"] == "defective"


def test_above_threshold_selects_correct_fault_type_via_argmax():
    # inner_race (index 1) has the highest probability.
    fault_proba = np.array([0.1, 0.7, 0.2])

    result = build_inspection_result(0.9, fault_proba, threshold=0.3, class_names=CLASS_NAMES)

    assert result["status"] == "defective"
    assert result["fault_type"] == "inner_race"
    assert result["fault_confidence"] == 0.7


def test_fault_type_selection_not_hardcoded_to_first_class():
    """confirms the selection depends on which
    class has the highest probability, not always returning index 0."""
    fault_proba = np.array([0.05, 0.05, 0.9])  # ball (index 2) dominant

    result = build_inspection_result(0.9, fault_proba, threshold=0.3, class_names=CLASS_NAMES)

    assert result["fault_type"] == "ball"