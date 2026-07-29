
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from defect_detection.evaluation.visualization import (
    plot_defect_gate_confusion_matrix,
    plot_fault_type_confusion_matrix,
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