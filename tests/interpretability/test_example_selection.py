import numpy as np

from defect_detection.interpretability.example_selection import (
    find_defect_gate_examples,
    find_fault_type_examples,
)

CLASS_NAMES = ["outer_race", "inner_race", "ball"]


def test_defect_gate_cases_correctly_identified():
    predictions = {
        "is_defect_true": np.array([0, 0, 1, 1, 1]),
        "is_defect_pred": np.array([0, 1, 1, 0, 1]),
    }

    result = find_defect_gate_examples(predictions, n=3)

    assert result["correct_defective"] == [2, 4]
    assert result["false_negative"] == [3]
    assert result["false_positive"] == [1]


def test_defect_gate_returns_empty_list_when_case_absent():
    predictions = {
        "is_defect_true": np.array([0, 0, 1, 1]),
        "is_defect_pred": np.array([0, 0, 1, 1]),
    }

    result = find_defect_gate_examples(predictions)

    assert result["false_negative"] == []
    assert result["false_positive"] == []


def test_defect_gate_respects_n_limit():
    predictions = {
        "is_defect_true": np.array([1, 1, 1, 1, 1]),
        "is_defect_pred": np.array([1, 1, 1, 1, 1]),
    }

    result = find_defect_gate_examples(predictions, n=2)

    assert result["correct_defective"] == [0, 1]


def test_row_index_maps_to_global_position_not_local():
    """row_index must refer to the position in the full test set, not the
    position within the defective-only arrays."""
    predictions = {
        "is_defect_true": np.array([0, 0, 1, 1, 1]),
        "fault_class_true": np.array([0, 1, 2]),
        "fault_class_pred": np.array([0, 1, 2]),
    }

    result = find_fault_type_examples(predictions, CLASS_NAMES)

    assert result["outer_race"]["correct"][0]["row_index"] == 2
    assert result["inner_race"]["correct"][0]["row_index"] == 3
    assert result["ball"]["correct"][0]["row_index"] == 4


def test_predicted_class_captured_correctly():
    predictions = {
        "is_defect_true": np.array([1, 1]),
        "fault_class_true": np.array([0, 0]),
        "fault_class_pred": np.array([0, 1]),
    }

    result = find_fault_type_examples(predictions, CLASS_NAMES)

    assert result["outer_race"]["correct"][0]["predicted_class"] == 0
    assert result["outer_race"]["misclassified"][0]["predicted_class"] == 1


def test_misclassified_entries_have_different_predicted_class():
    predictions = {
        "is_defect_true": np.array([1, 1, 1]),
        "fault_class_true": np.array([0, 0, 1]),
        "fault_class_pred": np.array([1, 0, 1]),
    }

    result = find_fault_type_examples(predictions, CLASS_NAMES)

    misclassified = result["outer_race"]["misclassified"]
    assert len(misclassified) == 1
    assert misclassified[0]["predicted_class"] != 0


def test_returns_empty_list_when_no_misclassified_example_exists():
    predictions = {
        "is_defect_true": np.array([1, 1]),
        "fault_class_true": np.array([2, 2]),
        "fault_class_pred": np.array([2, 2]),
    }

    result = find_fault_type_examples(predictions, CLASS_NAMES)

    assert result["ball"]["misclassified"] == []
    assert len(result["ball"]["correct"]) == 2


def test_returns_empty_list_for_class_never_present():
    predictions = {
        "is_defect_true": np.array([1, 1]),
        "fault_class_true": np.array([0, 0]),
        "fault_class_pred": np.array([0, 0]),
    }

    result = find_fault_type_examples(predictions, CLASS_NAMES)

    assert result["inner_race"]["correct"] == []
    assert result["inner_race"]["misclassified"] == []


def test_fault_type_respects_n_limit():
    predictions = {
        "is_defect_true": np.array([1, 1, 1, 1]),
        "fault_class_true": np.array([0, 0, 0, 0]),
        "fault_class_pred": np.array([0, 0, 0, 0]),
    }

    result = find_fault_type_examples(predictions, CLASS_NAMES, n=2)

    assert len(result["outer_race"]["correct"]) == 2