
import numpy as np
import pytest
import torch

from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.models.fusion_model import MultimodalDefectClassifier
from tests.factories import make_synthetic_loader


def test_returns_expected_keys():
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=8, n_defective=3)

    result = collect_test_predictions(model, loader, torch.device("cpu"))

    assert set(result.keys()) == {
        "is_defect_true", "is_defect_pred", "defect_proba",
        "fault_class_true", "fault_class_pred",
    }


def test_array_lengths_match_full_dataset_across_multiple_batches():
    """Array lengths should equal the full dataset size, correctly aggregated
    across multiple DataLoader batches."""
    n_samples = 25
    batch_size = 8  # forces 4 batches (8, 8, 8, 1)
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=n_samples, n_defective=10, batch_size=batch_size)

    result = collect_test_predictions(model, loader, torch.device("cpu"))

    assert len(result["is_defect_true"]) == n_samples
    assert len(result["is_defect_pred"]) == n_samples
    assert len(result["defect_proba"]) == n_samples


def test_fault_class_arrays_length_matches_total_defective_count():
    """fault_class arrays should have length equal to the number of defective
    samples, not the full dataset size."""
    n_samples = 25
    n_defective = 10
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=n_samples, n_defective=n_defective, batch_size=8)

    result = collect_test_predictions(model, loader, torch.device("cpu"))

    assert len(result["fault_class_true"]) == n_defective
    assert len(result["fault_class_pred"]) == n_defective


def test_is_defect_pred_matches_thresholded_proba():
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=8, n_defective=3)
    threshold = 0.5

    result = collect_test_predictions(model, loader, torch.device("cpu"), defect_threshold=threshold)

    expected_pred = (result["defect_proba"] >= threshold).astype(int)
    assert np.array_equal(result["is_defect_pred"], expected_pred)


def test_threshold_zero_predicts_all_defective():
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=8, n_defective=3)

    result = collect_test_predictions(model, loader, torch.device("cpu"), defect_threshold=0.0)

    assert np.all(result["is_defect_pred"] == 1)


def test_threshold_above_one_predicts_all_normal():
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=8, n_defective=3)

    result = collect_test_predictions(model, loader, torch.device("cpu"), defect_threshold=1.01)

    assert np.all(result["is_defect_pred"] == 0)


def test_handles_zero_defective_samples():
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=8, n_defective=0)

    result = collect_test_predictions(model, loader, torch.device("cpu"))

    assert len(result["fault_class_true"]) == 0
    assert len(result["fault_class_pred"]) == 0
    assert np.issubdtype(result["fault_class_true"].dtype, np.integer)



def test_does_not_change_model_weights():
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=8, n_defective=3)

    param_before = next(model.fusion_mlp.parameters()).clone()
    collect_test_predictions(model, loader, torch.device("cpu"))
    param_after = next(model.fusion_mlp.parameters())

    assert torch.allclose(param_before, param_after)


def test_is_deterministic():
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=8, n_defective=3)

    result_1 = collect_test_predictions(model, loader, torch.device("cpu"))
    result_2 = collect_test_predictions(model, loader, torch.device("cpu"))

    assert np.array_equal(result_1["defect_proba"], result_2["defect_proba"])


@pytest.mark.parametrize("modality", ["both", "image", "vibration"])
def test_works_for_all_modalities(modality):
    model = MultimodalDefectClassifier(modality=modality)
    loader = make_synthetic_loader(n_samples=8, n_defective=3)

    result = collect_test_predictions(model, loader, torch.device("cpu"))

    assert len(result["is_defect_true"]) == 8