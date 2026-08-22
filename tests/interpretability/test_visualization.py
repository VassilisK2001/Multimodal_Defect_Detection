import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import shap

from defect_detection.data.features import FEATURE_NAMES
from defect_detection.interpretability.visualization import (
    denormalize_image,
    plot_defect_gate_gradcam_grid,
    plot_fault_type_gradcam_grid,
    plot_gradcam_overlay,
    plot_beeswarm_comparison,
    plot_dependence,
    plot_waterfall,
    plot_branch_contribution_scatter,
    plot_branch_contribution_violin,
)
from defect_detection.models.fusion_model import MultimodalDefectClassifier

CLASS_NAMES = ["outer_race", "inner_race", "ball"]

@pytest.fixture(autouse=True)
def close_figures_after_test():
    yield
    plt.close("all")

class _FakeDataset:
    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        vib_features = torch.randn(5)
        is_defect = torch.tensor(1.0)
        fault_class_idx = torch.tensor(0)
        area_ratio = torch.tensor(0.0)
        return image, vib_features, is_defect, fault_class_idx, area_ratio


def _make_explanation(n_samples: int = 10, seed: int = 0) -> shap.Explanation:
    rng = np.random.default_rng(seed)
    return shap.Explanation(
        values=rng.normal(size=(n_samples, len(FEATURE_NAMES))),
        base_values=np.full(n_samples, 0.5),
        data=rng.normal(size=(n_samples, len(FEATURE_NAMES))),
        feature_names=FEATURE_NAMES,
    )


def test_denormalize_inverts_normalization_correctly():
    raw = torch.full((3, 4, 4), 0.5)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = (raw - mean) / std

    recovered = denormalize_image(normalized)

    assert np.allclose(recovered, 0.5, atol=1e-5)


def test_denormalize_clamps_out_of_range_values():
    extreme = torch.full((3, 4, 4), 100.0)

    recovered = denormalize_image(extreme)

    assert recovered.min() >= 0.0
    assert recovered.max() <= 1.0


def test_overlay_returns_figure_when_no_ax_given():
    image = np.random.rand(10, 10, 3)
    heatmap = np.random.rand(10, 10)

    fig = plot_gradcam_overlay(image, heatmap)

    assert isinstance(fig, plt.Figure)


def test_overlay_returns_none_and_draws_on_given_ax():
    image = np.random.rand(10, 10, 3)
    heatmap = np.random.rand(10, 10)
    fig, ax = plt.subplots()

    result = plot_gradcam_overlay(image, heatmap, ax=ax)

    assert result is None
    assert len(ax.images) == 2


def test_defect_gate_grid_produces_correct_panel_count():
    model = MultimodalDefectClassifier(modality="image")
    dataset = _FakeDataset()
    defect_examples = {"correct_defective": [0], "false_negative": [1], "false_positive": [2]}

    fig = plot_defect_gate_gradcam_grid(model, dataset, defect_examples, n=1)

    assert len(fig.axes) == 3


def test_defect_gate_grid_handles_missing_examples():
    model = MultimodalDefectClassifier(modality="image")
    dataset = _FakeDataset()
    defect_examples = {"correct_defective": [0, 1], "false_negative": [], "false_positive": [2]}

    fig = plot_defect_gate_gradcam_grid(model, dataset, defect_examples, n=2)

    assert len(fig.axes) == 6
    titles = [ax.get_title() for ax in fig.axes]
    assert sum("no example found" in title for title in titles) == 3


def test_fault_type_grid_produces_correct_panel_count():
    model = MultimodalDefectClassifier(modality="image")
    dataset = _FakeDataset()
    fault_examples = {
        name: {
            "correct": [{"row_index": 0, "predicted_class": 0}],
            "misclassified": [{"row_index": 1, "predicted_class": 1}],
        }
        for name in CLASS_NAMES
    }

    fig = plot_fault_type_gradcam_grid(model, dataset, fault_examples, CLASS_NAMES, n=1)

    assert len(fig.axes) == 2 * len(CLASS_NAMES)


def test_fault_type_grid_handles_missing_examples():
    model = MultimodalDefectClassifier(modality="image")
    dataset = _FakeDataset()
    fault_examples = {
        "outer_race": {"correct": [{"row_index": 0, "predicted_class": 0}], "misclassified": []},
        "inner_race": {"correct": [], "misclassified": []},
        "ball": {"correct": [{"row_index": 0, "predicted_class": 2}], "misclassified": []},
    }

    fig = plot_fault_type_gradcam_grid(model, dataset, fault_examples, CLASS_NAMES, n=1)

    titles = [ax.get_title() for ax in fig.axes]
    assert sum("no example found" in title for title in titles) == 4

 
def test_beeswarm_subplots_correctly_titled():
    vib_values = _make_explanation(seed=1)
    fusion_values = _make_explanation(seed=2)
 
    fig = plot_beeswarm_comparison(vib_values, fusion_values, title="Head 1 Comparison")
 
    ax_vib, ax_fusion = fig.axes[0], fig.axes[1]
    assert ax_vib.get_title() == "Vibration-only"
    assert ax_fusion.get_title() == "Fusion (both)"
    assert getattr(fig, "_suptitle").get_text() == "Head 1 Comparison"
 
 
def test_beeswarm_subplots_contain_distinct_rendered_content():
    """Guards against a missing/misapplied plt.sca() causing both summary_plot
    calls to draw into the same axes instead of their intended separate ones."""
    vib_values = _make_explanation(seed=1)
    fusion_values = _make_explanation(seed=2)
 
    fig = plot_beeswarm_comparison(vib_values, fusion_values)
 
    ax_vib, ax_fusion = fig.axes[0], fig.axes[1]
    assert ax_vib.has_data()
    assert ax_fusion.has_data()
 
  
def test_dependence_plots_the_requested_feature():
    shap_values = _make_explanation()
 
    fig = plot_dependence(shap_values, feature_name="Crest Factor", title="Crest Factor dependence")
 
    ax = fig.axes[0]
    assert "Crest Factor" in ax.get_xlabel()
 
 
def test_dependence_sets_the_given_title():
    shap_values = _make_explanation()
 
    fig = plot_dependence(shap_values, feature_name="RMS", title="RMS dependence")
 
    assert fig.axes[0].get_title() == "RMS dependence"
 
  
def test_waterfall_returns_figure_for_valid_row_index():
    shap_values = _make_explanation(n_samples=5)
 
    fig = plot_waterfall(shap_values, row_index=2, title="row 2")
 
    assert isinstance(fig, plt.Figure)
    assert getattr(fig, "_suptitle").get_text() == "row 2"
 
 
def test_waterfall_raises_for_out_of_range_row_index():
    shap_values = _make_explanation(n_samples=5)
 
    with pytest.raises(IndexError):
        plot_waterfall(shap_values, row_index=99)

 
def test_scatter_correctly_separates_points_by_label():
    phi_image = np.array([0.1, 0.2, -0.1, -0.2, 0.3])
    phi_vib = np.array([0.5, 0.6, -0.5, -0.6, 0.1])
    labels = np.array(["a", "a", "b", "b", "a"])
 
    fig = plot_branch_contribution_scatter(phi_image, phi_vib, labels)
 
    scatter_collections = {
        coll.get_label(): coll for coll in fig.axes[0].collections
        if coll.get_label() in ("a", "b")
    }
    assert set(scatter_collections.keys()) == {"a", "b"}
 
    offsets_a = scatter_collections["a"].get_offsets()
    expected_a = np.column_stack([phi_image[labels == "a"], phi_vib[labels == "a"]])
    assert np.allclose(sorted(offsets_a.tolist()), sorted(expected_a.tolist()))
 
 
def test_scatter_includes_equal_contribution_reference_line():
    phi_image = np.array([0.1, 0.2, 0.3])
    phi_vib = np.array([0.1, 0.2, 0.3])
    labels = np.array(["a", "a", "a"])
 
    fig = plot_branch_contribution_scatter(phi_image, phi_vib, labels)
 
    reference_lines = [line for line in fig.axes[0].get_lines() if line.get_linestyle() == "--"]
    assert len(reference_lines) >= 1
    x_data, y_data = reference_lines[0].get_xdata(), reference_lines[0].get_ydata()
    assert np.allclose(x_data, y_data)  # slope-1 diagonal
 
  
def test_violin_produces_correct_number_of_subplots_and_titles():
    class_names = ["outer_race", "inner_race", "ball"]
    phi_image_per_class = {name: np.random.randn(10) for name in class_names}
    phi_vib_per_class = {name: np.random.randn(10) for name in class_names}
 
    fig = plot_branch_contribution_violin(phi_image_per_class, phi_vib_per_class, class_names)
 
    assert len(fig.axes) == 3
    assert [ax.get_title() for ax in fig.axes] == class_names
 
 
def test_violin_handles_single_class_without_crashing():
    """Regression test for the len(class_names) == 1 edge case."""
    class_names = ["ball"]
    phi_image_per_class = {"ball": np.random.randn(10)}
    phi_vib_per_class = {"ball": np.random.randn(10)}
 
    fig = plot_branch_contribution_violin(phi_image_per_class, phi_vib_per_class, class_names)
 
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "ball"
 
 