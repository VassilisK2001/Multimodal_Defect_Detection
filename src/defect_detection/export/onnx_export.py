import time
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from defect_detection.models.fusion_model import MultimodalDefectClassifier


class FusionModelWithActivations(nn.Module):
    """Wraps the fusion model so sigmoid/softmax are baked into the exported
    ONNX graph."""

    def __init__(self, model: MultimodalDefectClassifier):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, vib_features: torch.Tensor):
        defect_logit, fault_logits = self.model(image=image, vib_features=vib_features)
        defect_proba = torch.sigmoid(defect_logit)
        fault_proba = torch.softmax(fault_logits, dim=1)
        return defect_proba, fault_proba


def export_model_to_onnx(wrapped_model: FusionModelWithActivations, sample_image: torch.Tensor,
                          sample_vib_features: torch.Tensor, output_path, opset_version: int = 17) -> None:
    """Trace and export wrapped_model to a single ONNX graph with dynamic
    batch axes on all inputs/outputs.

    Args:
        wrapped_model: A FusionModelWithActivations instance, in eval mode.
        sample_image: (1, 3, H, W) real image tensor, used for tracing.
        sample_vib_features: (1, 5) real normalized vibration tensor, used
            for tracing.
        output_path: Where to write the .onnx file.
        opset_version: ONNX opset version.
    """
    torch.onnx.export(
        wrapped_model,
        (sample_image, sample_vib_features),
        str(output_path),
        input_names=["image", "vib_features"],
        output_names=["defect_proba", "fault_proba"],
        dynamic_axes={
            "image": {0: "batch"}, "vib_features": {0: "batch"},
            "defect_proba": {0: "batch"}, "fault_proba": {0: "batch"},
        },
        opset_version=opset_version,
        dynamo=False,
    )


def run_parity_check(wrapped_model: FusionModelWithActivations, onnx_session: ort.InferenceSession,
                      image: torch.Tensor, vib_features: torch.Tensor, atol: float = 1e-4) -> dict:
    """Compare the exported ONNX model's probability outputs against the
    original PyTorch model's, on the same real input.

    Args:
        wrapped_model: The traced FusionModelWithActivations instance.
        onnx_session: An InferenceSession loaded from the exported .onnx file.
        image: (1, 3, H, W) real image tensor.
        vib_features: (1, 5) real normalized vibration tensor.
        atol: Absolute tolerance for the comparison.

    Returns:
        Dict with 'defect_matches', 'fault_matches', 'passed' (both booleans
        combined).
    """
    with torch.no_grad():
        torch_defect_proba, torch_fault_proba = wrapped_model(image, vib_features)

    onnx_defect_proba, onnx_fault_proba = onnx_session.run(
        ["defect_proba", "fault_proba"], {"image": image.numpy(), "vib_features": vib_features.numpy()},
    )

    defect_matches = bool(np.allclose(torch_defect_proba.numpy(), onnx_defect_proba, atol=atol))
    fault_matches = bool(np.allclose(torch_fault_proba.numpy(), onnx_fault_proba, atol=atol))

    return {
        "defect_matches": defect_matches, "fault_matches": fault_matches,
        "passed": defect_matches and fault_matches,
    }


def make_torch_runner(wrapped_model: FusionModelWithActivations, image: torch.Tensor,
                       vib_features: torch.Tensor) -> Callable[[], None]:
    """Build a zero-argument callable running one PyTorch forward pass, for
    use with benchmark_latency()."""
    def _run() -> None:
        with torch.no_grad():
            wrapped_model(image, vib_features)
    return _run


def make_onnx_runner(onnx_session: ort.InferenceSession, image: torch.Tensor,
                      vib_features: torch.Tensor) -> Callable[[], None]:
    """Build a zero-argument callable running one ONNX Runtime inference
    call, for use with benchmark_latency()."""
    def _run() -> None:
        onnx_session.run(
            ["defect_proba", "fault_proba"], {"image": image.numpy(), "vib_features": vib_features.numpy()},
        )
    return _run


def benchmark_latency(fn: Callable[[], None], n_runs: int, n_warmup: int) -> dict:
    """Time n_runs calls to fn(), after n_warmup untimed warmup calls.

    Args:
        fn: A zero-argument callable to time.
        n_runs: Number of timed calls.
        n_warmup: Number of untimed warmup calls, excluded from the timing.

    Returns:
        Dict with 'mean_ms', 'std_ms' latency in milliseconds.
    """
    for _ in range(n_warmup):
        fn()

    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)

    timings_arr = np.array(timings)
    return {"mean_ms": float(timings_arr.mean()), "std_ms": float(timings_arr.std())}