import sys
from pathlib import Path

import numpy as np
import pytest
import torch

DEPLOYMENT_DIR = Path(__file__).resolve().parents[3] / "deployment"
sys.path.insert(0, str(DEPLOYMENT_DIR))

from inference.onnx_model import OnnxFusionModel  

from defect_detection.export.onnx_export import FusionModelWithActivations, export_model_to_onnx
from defect_detection.models.fusion_model import MultimodalDefectClassifier


@pytest.fixture
def onnx_path(tmp_path) -> Path:
    model = MultimodalDefectClassifier(modality="both").eval()
    wrapped = FusionModelWithActivations(model).eval()
    image = torch.randn(1, 3, 224, 224)
    vib = torch.randn(1, 5)
    path = tmp_path / "model.onnx"
    export_model_to_onnx(wrapped, image, vib, path)
    return path


def test_predict_returns_correct_types_and_shapes(onnx_path):
    model = OnnxFusionModel(onnx_path)
    image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    vib = np.random.randn(1, 5).astype(np.float32)

    defect_proba, fault_proba = model.predict(image, vib)

    assert isinstance(defect_proba, float)
    assert fault_proba.shape == (3,)


def test_predict_returns_valid_probability_ranges(onnx_path):
    model = OnnxFusionModel(onnx_path)
    image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    vib = np.random.randn(1, 5).astype(np.float32)

    defect_proba, fault_proba = model.predict(image, vib)

    assert 0.0 <= defect_proba <= 1.0
    assert np.allclose(fault_proba.sum(), 1.0, atol=1e-5)


def test_predict_deterministic_for_same_input(onnx_path):
    model = OnnxFusionModel(onnx_path)
    image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    vib = np.random.randn(1, 5).astype(np.float32)

    defect_a, fault_a = model.predict(image, vib)
    defect_b, fault_b = model.predict(image, vib)

    assert defect_a == pytest.approx(defect_b)
    assert np.allclose(fault_a, fault_b)


def test_predict_different_inputs_give_different_outputs(onnx_path):
    model = OnnxFusionModel(onnx_path)
    image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    vib_a = np.zeros((1, 5), dtype=np.float32)
    vib_b = np.ones((1, 5), dtype=np.float32) * 5.0

    defect_a, _ = model.predict(image, vib_a)
    defect_b, _ = model.predict(image, vib_b)

    assert defect_a != pytest.approx(defect_b)


def test_custom_intra_op_num_threads_does_not_break_prediction(onnx_path):
    model = OnnxFusionModel(onnx_path, intra_op_num_threads=1)
    image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    vib = np.random.randn(1, 5).astype(np.float32)

    defect_proba, fault_proba = model.predict(image, vib)

    assert isinstance(defect_proba, float)
    assert fault_proba.shape == (3,)