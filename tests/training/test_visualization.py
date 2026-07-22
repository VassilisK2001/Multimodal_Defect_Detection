
import math

import matplotlib.pyplot as plt
import pytest

from defect_detection.training.visualization import (
    plot_defect_accuracy_curve,
    plot_defect_loss_curve,
    plot_fault_type_accuracy_curve,
    plot_fault_type_loss_curve,
)

@pytest.fixture(autouse=True)
def close_figures_after_test():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_history() -> dict:
    n_epochs = 20
    return {
        "train_defect_loss": [1.0 - 0.03 * i for i in range(n_epochs)],
        "val_defect_loss": [1.1 - 0.025 * i for i in range(n_epochs)],
        "train_fault_type_loss": [0.8 - 0.02 * i for i in range(n_epochs)],
        "val_fault_type_loss": [0.9 - 0.015 * i for i in range(n_epochs)],
        "train_defect_accuracy": [0.5 + 0.02 * i for i in range(n_epochs)],
        "val_defect_accuracy": [0.5 + 0.015 * i for i in range(n_epochs)],
        "train_fault_type_accuracy": [0.3 + 0.02 * i for i in range(n_epochs)],
        "val_fault_type_accuracy": [0.3 + 0.01 * i for i in range(n_epochs)],
    }


@pytest.mark.parametrize("plot_fn, keys", [
    (plot_defect_loss_curve, ("train_defect_loss", "val_defect_loss")),
    (plot_fault_type_loss_curve, ("train_fault_type_loss", "val_fault_type_loss")),
    (plot_defect_accuracy_curve, ("train_defect_accuracy", "val_defect_accuracy")),
    (plot_fault_type_accuracy_curve, ("train_fault_type_accuracy", "val_fault_type_accuracy")),
])
def test_returns_figure_with_two_lines(plot_fn, keys, sample_history):
    """Each plot function should return a Figure with exactly 2 lines (train, val)."""
    fig = plot_fn(sample_history)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes[0].lines) == 2


@pytest.mark.parametrize("plot_fn, keys", [
    (plot_defect_loss_curve, ("train_defect_loss", "val_defect_loss")),
    (plot_fault_type_loss_curve, ("train_fault_type_loss", "val_fault_type_loss")),
    (plot_defect_accuracy_curve, ("train_defect_accuracy", "val_defect_accuracy")),
    (plot_fault_type_accuracy_curve, ("train_fault_type_accuracy", "val_fault_type_accuracy")),
])
def test_plotted_data_matches_history_values(plot_fn, keys, sample_history):
    """The plotted y-data should match the corresponding history values."""
    fig = plot_fn(sample_history)
    lines = fig.axes[0].lines
    train_key, val_key = keys

    plotted_value_sets = [list(line.get_ydata()) for line in lines]
    assert list(sample_history[train_key]) in plotted_value_sets
    assert list(sample_history[val_key]) in plotted_value_sets


def test_defect_accuracy_plot_has_fixed_y_limits(sample_history):
    fig = plot_defect_accuracy_curve(sample_history)
    assert fig.axes[0].get_ylim() == (0.0, 1.0)


def test_fault_type_accuracy_plot_has_fixed_y_limits(sample_history):
    fig = plot_fault_type_accuracy_curve(sample_history)
    assert fig.axes[0].get_ylim() == (0.0, 1.0)


@pytest.mark.parametrize("plot_fn, keys", [
    (plot_fault_type_loss_curve, ("train_fault_type_loss", "val_fault_type_loss")),
    (plot_fault_type_accuracy_curve, ("train_fault_type_accuracy", "val_fault_type_accuracy")),
])
def test_handles_nan_values_without_error(plot_fn, keys, sample_history):
    """Plotting should not raise when fault-type loss/accuracy values are NaN."""
    train_key, val_key = keys
    history_with_nan = dict(sample_history)
    history_with_nan[train_key] = list(sample_history[train_key])
    history_with_nan[train_key][3] = math.nan
    history_with_nan[val_key] = list(sample_history[val_key])
    history_with_nan[val_key][7] = math.nan

    fig = plot_fn(history_with_nan)
    assert isinstance(fig, plt.Figure)


@pytest.mark.parametrize("plot_fn", [
    plot_defect_loss_curve, plot_fault_type_loss_curve,
    plot_defect_accuracy_curve, plot_fault_type_accuracy_curve,
])
def test_title_includes_modality(plot_fn, sample_history):
    fig = plot_fn(sample_history, modality="vibration")
    assert "vibration" in fig.axes[0].get_title()