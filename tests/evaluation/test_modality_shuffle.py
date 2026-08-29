
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from defect_detection.evaluation.modality_shuffle import (
    _corrupt_tensor,
    collect_predictions_with_corruption,
    corrupt_batch,
    run_modality_shuffle_test,
)
from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.models.fusion_model import MultimodalDefectClassifier
from tests.factories import make_synthetic_loader


def test_zero_method_produces_all_zeros():
    tensor = torch.randn(8, 5)
    result = _corrupt_tensor(tensor, "zero")
    assert torch.all(result == 0)


def test_shuffle_method_is_a_genuine_permutation():
    tensor = torch.arange(20).float().unsqueeze(1)  # distinct values per row
    result = _corrupt_tensor(tensor, "shuffle")

    assert torch.equal(result.sort(dim=0).values, tensor.sort(dim=0).values)  # same multiset
    assert not torch.equal(result, tensor)  # actually reordered


def test_shuffle_method_reproducible_with_same_seed():
    tensor = torch.arange(20).float().unsqueeze(1)

    result_a = _corrupt_tensor(tensor, "shuffle", seed=42)
    result_b = _corrupt_tensor(tensor, "shuffle", seed=42)

    assert torch.equal(result_a, result_b)


def test_corrupt_tensor_raises_on_unknown_method():
    with pytest.raises(ValueError):
        _corrupt_tensor(torch.randn(4, 5), "unknown_method")


def test_corrupt_batch_only_modifies_targeted_modality():
    """Corrupting 'image' must leave vib_features, is_defect, and fault_class_idx
    completely unchanged, and vice versa."""
    loader = make_synthetic_loader(n_samples=8, n_defective=3)
    batch = next(iter(loader))
    images, vib_features, is_defect, fault_class_idx, area_ratio = batch

    corrupted = corrupt_batch(batch, "image", method="zero")
    assert not torch.equal(corrupted[0], images)
    assert torch.equal(corrupted[1], vib_features)
    assert torch.equal(corrupted[2], is_defect)
    assert torch.equal(corrupted[3], fault_class_idx)

    corrupted = corrupt_batch(batch, "vibration", method="zero")
    assert torch.equal(corrupted[0], images)
    assert not torch.equal(corrupted[1], vib_features)
    assert torch.equal(corrupted[2], is_defect)
    assert torch.equal(corrupted[3], fault_class_idx)


def test_corrupt_batch_raises_on_unknown_modality():
    loader = make_synthetic_loader(n_samples=4, n_defective=1)
    batch = next(iter(loader))
    with pytest.raises(ValueError):
        corrupt_batch(batch, "audio", method="zero")


def test_collect_predictions_with_corruption_matches_baseline_shape():
    """Should return the same dict shape as collect_test_predictions."""
    
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=8, n_defective=3)
    device = torch.device("cpu")

    baseline = collect_test_predictions(model, loader, device)
    corrupted = collect_predictions_with_corruption(model, loader, device, "image", method="zero")

    assert set(baseline.keys()) == set(corrupted.keys())


def test_corrupt_modalities_controls_output_entries():
    """Requesting only 'image' should produce baseline + image_corrupted, not
    vibration_corrupted."""
    fake_predictions = {
        "is_defect_true": np.array([0, 1]), "is_defect_pred": np.array([0, 1]),
        "defect_proba": np.array([0.1, 0.9]),
        "fault_class_true": np.array([1]), "fault_class_pred": np.array([1]),
    }

    with patch("defect_detection.evaluation.modality_shuffle.MultimodalDefectDataset"), \
         patch("defect_detection.evaluation.modality_shuffle.DataLoader"), \
         patch("defect_detection.evaluation.modality_shuffle.collect_test_predictions",
               return_value=fake_predictions), \
         patch("defect_detection.evaluation.modality_shuffle.collect_predictions_with_corruption",
               return_value=fake_predictions):

        results = run_modality_shuffle_test(
            model=MagicMock(), test_df=pd.DataFrame(), window_size=2048, fs=12000,
            class_names=["outer_race", "inner_race", "ball"],
            vib_mean=np.zeros(5), vib_std=np.ones(5), device=torch.device("cpu"),
            corrupt_modalities=("image",),
        )

    assert set(results.keys()) == {"baseline", "image_corrupted"}