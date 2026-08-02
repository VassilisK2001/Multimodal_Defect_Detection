
import json

import matplotlib.pyplot as plt
import numpy as np

from defect_detection.evaluation.reporting import save_evaluation_results


def test_creates_expected_files(tmp_path):
    fig, _ = plt.subplots()
    result = {
        "defect_metrics": {"normal": {"f1": np.float64(0.9)}},
        "fault_metrics": {"macro_f1": np.float64(0.8)},
        "figures": {"defect_confusion_matrix": fig},
    }

    save_evaluation_results("both", result, tmp_path)

    assert (tmp_path / "both" / "metrics.json").exists()
    assert (tmp_path / "both" / "defect_confusion_matrix.png").exists()


def test_metrics_json_is_valid_json(tmp_path):
    fig, _ = plt.subplots()
    result = {
        "defect_metrics": {"normal": {"f1": np.float64(0.9)}},
        "fault_metrics": {"macro_f1": np.float64(0.8)},
        "figures": {"defect_confusion_matrix": fig},
    }

    save_evaluation_results("both", result, tmp_path)

    with open(tmp_path / "both" / "metrics.json") as f:
        json.load(f)  


def test_metrics_json_preserves_numeric_values(tmp_path):
    fig, _ = plt.subplots()
    result = {
        "defect_metrics": {"normal": {"f1": np.float64(0.9)}},
        "fault_metrics": {"macro_f1": np.float64(0.8)},
        "figures": {"defect_confusion_matrix": fig},
    }

    save_evaluation_results("both", result, tmp_path)

    with open(tmp_path / "both" / "metrics.json") as f:
        loaded = json.load(f)

    assert loaded["fault_metrics"]["macro_f1"] == 0.8
    assert loaded["defect_metrics"]["normal"]["f1"] == 0.9


def test_creates_parent_directories_if_missing(tmp_path):
    fig, _ = plt.subplots()
    result = {"defect_metrics": {}, "fault_metrics": {}, "figures": {"x": fig}}
    nested_output_dir = tmp_path / "a" / "b" / "c"

    save_evaluation_results("vibration", result, nested_output_dir)

    assert (nested_output_dir / "vibration" / "x.png").exists()