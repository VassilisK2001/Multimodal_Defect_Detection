
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from defect_detection.evaluation.visualization import (
    plot_defect_gate_confusion_matrix,
    plot_defect_gate_global_curves,
    plot_fault_type_confusion_matrix,
    plot_fault_type_global_pr_curves,
    plot_pr_curve_with_threshold,
)
 

@pytest.fixture(autouse=True)
def close_figures_after_test():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def _heatmap_values(fig: plt.Figure, n_rows: int) -> np.ndarray:
    """Extract the plotted 2D array from a seaborn heatmap's Figure."""
    ax = fig.axes[0]
    quadmesh = ax.collections[0]
    return quadmesh.get_array().reshape(n_rows, -1)


def test_defect_gate_confusion_matrix_returns_figure():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    fig = plot_defect_gate_confusion_matrix(y_true, y_pred)
    assert isinstance(fig, plt.Figure)


def test_defect_gate_confusion_matrix_values_match_actual_confusion_matrix():
    """The plotted heatmap values must match an independently computed confusion
    matrix."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0])
    expected_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig = plot_defect_gate_confusion_matrix(y_true, y_pred)
    plotted = _heatmap_values(fig, n_rows=2)

    assert np.array_equal(plotted, expected_cm)


def test_defect_gate_confusion_matrix_axis_labels():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    fig = plot_defect_gate_confusion_matrix(y_true, y_pred)
    ax = fig.axes[0]

    x_labels = [t.get_text() for t in ax.get_xticklabels()]
    y_labels = [t.get_text() for t in ax.get_yticklabels()]

    assert x_labels == ["normal", "defect"]
    assert y_labels == ["normal", "defect"]


def test_defect_gate_confusion_matrix_handles_missing_class():
    """Should not raise when one class never occurs in y_true or y_pred."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])

    fig = plot_defect_gate_confusion_matrix(y_true, y_pred)
    assert isinstance(fig, plt.Figure)


def test_fault_type_confusion_matrix_returns_figure():
    class_names = ["outer_race", "inner_race", "ball"]
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])

    fig = plot_fault_type_confusion_matrix(y_true, y_pred, class_names)
    assert isinstance(fig, plt.Figure)


def test_fault_type_confusion_matrix_values_match_actual_confusion_matrix():
    class_names = ["outer_race", "inner_race", "ball"]
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    expected_cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig = plot_fault_type_confusion_matrix(y_true, y_pred, class_names)
    plotted = _heatmap_values(fig, n_rows=3)

    assert np.array_equal(plotted, expected_cm)


def test_fault_type_confusion_matrix_axis_labels_match_class_order():
    class_names = ["outer_race", "inner_race", "ball"]
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])

    fig = plot_fault_type_confusion_matrix(y_true, y_pred, class_names)
    ax = fig.axes[0]

    x_labels = [t.get_text() for t in ax.get_xticklabels()]
    y_labels = [t.get_text() for t in ax.get_yticklabels()]

    assert x_labels == class_names
    assert y_labels == class_names


def test_fault_type_confusion_matrix_handles_missing_class():
    """Should not raise when one fault type never occurs in this particular split."""
    class_names = ["outer_race", "inner_race", "ball"]
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 1, 1])

    fig = plot_fault_type_confusion_matrix(y_true, y_pred, class_names)
    assert isinstance(fig, plt.Figure)

 
def _fake_oof_results_defect(is_defect_true: np.ndarray, model_probas: dict) -> dict:
    return {
        "is_defect_true": is_defect_true,
        "models": {name: {"oof_defect_proba": proba} for name, proba in model_probas.items()},
    }
 
 
def _fake_global_metrics_defect(model_probas: dict) -> dict:
    return {name: {"defect": {"roc_auc": 0.9, "pr_auc": 0.8}} for name in model_probas}
 
 
def test_defect_gate_curves_returns_two_subplots_with_correct_line_counts():
    is_defect_true = np.array([0, 0, 1, 1])
    model_probas = {
        "image": np.array([0.2, 0.3, 0.7, 0.8]),
        "vibration": np.array([0.1, 0.4, 0.6, 0.9]),
    }
    oof_results = _fake_oof_results_defect(is_defect_true, model_probas)
    global_metrics = _fake_global_metrics_defect(model_probas)
 
    fig = plot_defect_gate_global_curves(oof_results, global_metrics)
 
    assert len(fig.axes) == 2
    assert len(fig.axes[0].lines) == 3
    assert len(fig.axes[1].lines) == 3
 
 
def test_defect_gate_curves_roc_diagonal_is_identity_line():
    is_defect_true = np.array([0, 0, 1, 1])
    model_probas = {"image": np.array([0.2, 0.3, 0.7, 0.8])}
    oof_results = _fake_oof_results_defect(is_defect_true, model_probas)
    global_metrics = _fake_global_metrics_defect(model_probas)
 
    fig = plot_defect_gate_global_curves(oof_results, global_metrics)
 
    diagonal_line = fig.axes[0].lines[-1]
    assert list(diagonal_line.get_xdata()) == [0, 1]
    assert list(diagonal_line.get_ydata()) == [0, 1]
 
 
def test_defect_gate_curves_pr_baseline_matches_actual_defect_rate():
    is_defect_true = np.array([0, 0, 0, 1])  
    model_probas = {"image": np.array([0.2, 0.3, 0.1, 0.8])}
    oof_results = _fake_oof_results_defect(is_defect_true, model_probas)
    global_metrics = _fake_global_metrics_defect(model_probas)
 
    fig = plot_defect_gate_global_curves(oof_results, global_metrics)
 
    ax_pr = fig.axes[1]
    baseline_lines = [line for line in ax_pr.get_lines() if line.get_linestyle() == "--"]
    assert len(baseline_lines) == 1
    assert np.allclose(baseline_lines[0].get_ydata(), 0.25)
 
 
def test_defect_gate_curves_legend_labels_include_correct_auc_values():
    is_defect_true = np.array([0, 0, 1, 1])
    model_probas = {"image": np.array([0.2, 0.3, 0.7, 0.8])}
    oof_results = _fake_oof_results_defect(is_defect_true, model_probas)
    global_metrics = {"image": {"defect": {"roc_auc": 0.777, "pr_auc": 0.666}}}
 
    fig = plot_defect_gate_global_curves(oof_results, global_metrics)
 
    roc_legend_text = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    pr_legend_text = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
 
    assert any("0.777" in text for text in roc_legend_text)
    assert any("0.666" in text for text in pr_legend_text)
 
  
def test_fault_type_curves_subplot_count_and_order_match_request():
    class_names = ["outer_race", "inner_race", "ball"]
    is_defect_true = np.array([0, 1, 1, 1])
    fault_class_true = np.array([-1, 0, 1, 2])
    oof_results = {
        "is_defect_true": is_defect_true,
        "fault_class_true": fault_class_true,
        "models": {
            "image": {"oof_fault_proba": np.array([
                [0, 0, 0], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
            ])},
        },
    }
    global_metrics = {
        "image": {"fault": {name: {"roc_auc": 0.9, "pr_auc": 0.8} for name in class_names}},
    }
 
    fig = plot_fault_type_global_pr_curves(
        oof_results, global_metrics, class_names, subplot_order=["ball", "outer_race", "inner_race"],
    )
 
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["ball", "outer_race", "inner_race"]
 
 
def test_fault_type_curves_defaults_to_class_names_order_when_unspecified():
    class_names = ["outer_race", "inner_race", "ball"]
    oof_results = {
        "is_defect_true": np.array([0, 1, 1, 1]),
        "fault_class_true": np.array([-1, 0, 1, 2]),
        "models": {
            "image": {"oof_fault_proba": np.array([
                [0, 0, 0], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
            ])},
        },
    }
    global_metrics = {
        "image": {"fault": {name: {"roc_auc": 0.9, "pr_auc": 0.8} for name in class_names}},
    }
 
    fig = plot_fault_type_global_pr_curves(oof_results, global_metrics, class_names)
 
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == class_names
 
 
def test_fault_type_curves_use_only_defective_masked_samples():
    """Curves are computed from defective rows only, not normal rows."""
    class_names = ["outer_race", "inner_race", "ball"]
    is_defect_true = np.array([0, 0, 1, 1, 1])
    fault_class_true = np.array([-1, -1, 0, 1, 2])
    oof_results = {
        "is_defect_true": is_defect_true,
        "fault_class_true": fault_class_true,
        "models": {
            "image": {"oof_fault_proba": np.array([
                [0, 0, 0], [0, 0, 0],
                [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9],
            ])},
        },
    }
    global_metrics = {"image": {"fault": {name: {"roc_auc": 1.0, "pr_auc": 1.0} for name in class_names}}}
 
    fig = plot_fault_type_global_pr_curves(oof_results, global_metrics, class_names)
 
    outer_race_ax = fig.axes[class_names.index("outer_race")]
    model_line = outer_race_ax.lines[0]
    assert np.max(model_line.get_ydata()) == pytest.approx(1.0)


def test_pr_curve_marks_the_given_operating_point_exactly():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.4, 0.6, 0.8])
 
    fig = plot_pr_curve_with_threshold(
        y_true, y_proba, threshold=0.5, precision_at_threshold=0.75, recall_at_threshold=0.6,
    )
 
    scatter = next(c for c in fig.axes[0].collections)
    offset = scatter.get_offsets()[0]
    assert offset[0] == pytest.approx(0.6)
    assert offset[1] == pytest.approx(0.75)
 
 
def test_pr_curve_baseline_matches_defect_rate():
    is_defect_true = np.array([0, 0, 0, 1])
    y_proba = np.array([0.2, 0.3, 0.1, 0.8])
 
    fig = plot_pr_curve_with_threshold(
        is_defect_true, y_proba, threshold=0.5, precision_at_threshold=1.0, recall_at_threshold=1.0,
    )
 
    baseline_lines = [line for line in fig.axes[0].get_lines() if line.get_linestyle() == "--"]
    assert len(baseline_lines) == 1
    assert np.allclose(baseline_lines[0].get_ydata(), 0.25)
 
 
def test_pr_curve_returns_figure_with_curve_and_baseline_lines():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.4, 0.6, 0.8])
 
    fig = plot_pr_curve_with_threshold(
        y_true, y_proba, threshold=0.5, precision_at_threshold=0.75, recall_at_threshold=0.6,
    )
 
    assert isinstance(fig, plt.Figure)
    solid_lines = [line for line in fig.axes[0].get_lines() if line.get_linestyle() == "-"]
    dashed_lines = [line for line in fig.axes[0].get_lines() if line.get_linestyle() == "--"]
    assert len(solid_lines) >= 1
    assert len(dashed_lines) == 1
 
 