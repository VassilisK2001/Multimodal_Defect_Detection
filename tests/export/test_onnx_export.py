"""
tests/export/test_onnx_export.py

Tests for src/defect_detection/export/onnx_export.py.
"""

import time
from unittest.mock import MagicMock

import numpy as np
import onnxruntime as ort
import pytest
import torch

from defect_detection.export.onnx_export import (
    FusionModelWithActivations,
    benchmark_latency,
    export_model_to_onnx,
    make_onnx_runner,
    make_torch_runner,
    run_parity_check,
)
from defect_detection.models.fusion_model import MultimodalDefectClassifier


@pytest.fixture
def wrapped_model() -> FusionModelWithActivations:
    model = MultimodalDefectClassifier(modality="both").eval()
    return FusionModelWithActivations(model).eval()


@pytest.fixture
def sample_inputs():
    return torch.randn(1, 3, 224, 224), torch.randn(1, 5)


# --- FusionModelWithActivations -----------------------------------------------------------

def test_wrapper_applies_sigmoid_and_softmax(wrapped_model, sample_inputs):
    image, vib = sample_inputs

    defect_proba, fault_proba = wrapped_model(image, vib)

    assert torch.all(defect_proba >= 0) and torch.all(defect_proba <= 1)
    assert torch.allclose(fault_proba.sum(dim=1), torch.ones(1), atol=1e-5)


# --- export_model_to_onnx ---------------------------------------------------------------

def test_exported_model_handles_a_different_batch_size_than_tracing(wrapped_model, sample_inputs, tmp_path):
    """Verifies dynamic_axes allows a batch size different from the one used
    for tracing."""
    image, vib = sample_inputs
    onnx_path = tmp_path / "model.onnx"

    export_model_to_onnx(wrapped_model, image, vib, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    batch3_image = torch.randn(3, 3, 224, 224).numpy()
    batch3_vib = torch.randn(3, 5).numpy()
    defect_proba, fault_proba = session.run(
        ["defect_proba", "fault_proba"], {"image": batch3_image, "vib_features": batch3_vib},
    )

    assert defect_proba.shape[0] == 3
    assert fault_proba.shape[0] == 3


# --- run_parity_check ------------------------------------------------------------------

def test_parity_check_passes_when_outputs_genuinely_match(wrapped_model, sample_inputs, tmp_path):
    image, vib = sample_inputs
    onnx_path = tmp_path / "model.onnx"
    export_model_to_onnx(wrapped_model, image, vib, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    result = run_parity_check(wrapped_model, session, image, vib)

    assert result["passed"] is True
    assert result["defect_matches"] is True
    assert result["fault_matches"] is True


def test_parity_check_detects_a_real_mismatch(wrapped_model, sample_inputs):
    image, vib = sample_inputs
    fake_session = MagicMock()
    fake_session.run.return_value = (
        np.array([[0.999]], dtype=np.float32),
        np.array([[0.33, 0.33, 0.34]], dtype=np.float32),
    )

    result = run_parity_check(wrapped_model, fake_session, image, vib)

    assert result["passed"] is False


# --- make_torch_runner / make_onnx_runner ----------------------------------------------

def test_torch_runner_actually_invokes_the_model(wrapped_model, sample_inputs):
    image, vib = sample_inputs
    call_count = {"n": 0}
    original_forward = wrapped_model.forward

    def _counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return original_forward(*args, **kwargs)

    wrapped_model.forward = _counting_forward
    runner = make_torch_runner(wrapped_model, image, vib)

    runner()
    runner()

    assert call_count["n"] == 2


def test_onnx_runner_actually_invokes_the_session(sample_inputs):
    image, vib = sample_inputs
    fake_session = MagicMock()
    fake_session.run.return_value = (np.zeros((1, 1)), np.zeros((1, 3)))

    runner = make_onnx_runner(fake_session, image, vib)
    runner()
    runner()

    assert fake_session.run.call_count == 2


# --- benchmark_latency ------------------------------------------------------------------

def test_benchmark_excludes_warmup_calls_from_timing_but_still_makes_them():
    call_count = {"n": 0}

    def _fn():
        call_count["n"] += 1

    benchmark_latency(_fn, n_runs=5, n_warmup=3)

    assert call_count["n"] == 8


def test_benchmark_mean_roughly_matches_a_known_controllable_duration():
    sleep_seconds = 0.01

    def _fn():
        time.sleep(sleep_seconds)

    result = benchmark_latency(_fn, n_runs=10, n_warmup=2)

    assert result["mean_ms"] == pytest.approx(sleep_seconds * 1000, rel=0.5)