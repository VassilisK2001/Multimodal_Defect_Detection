import json
import logging

import onnxruntime as ort
import pandas as pd
import torch

from defect_detection.data.features import extract_raw_vib_features_from_df
from defect_detection.data.normalization import apply_vibration_normalization
from defect_detection.export.onnx_export import (
    FusionModelWithActivations,
    benchmark_latency,
    export_model_to_onnx,
    make_onnx_runner,
    make_torch_runner,
    run_parity_check,
)
from defect_detection.interpretability.shap_explain import load_images_for_df
from defect_detection.mlflow_utils import load_model_and_stats
from defect_detection.utils import find_project_root, load_yaml_config


logger = logging.getLogger(__name__)

N_BENCHMARK_RUNS = 50
N_WARMUP_RUNS = 5


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")
    test_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "test.csv")
    window_size, fs = data_config["window_size"], data_config["cwru"]["sampling_rate_hz"]
    output_dir = project_root / "deployment" / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading final deployed model ('both')...")
    model, vib_mean, vib_std = load_model_and_stats("both", device=torch.device("cpu"))
    model.eval()
    wrapped_model = FusionModelWithActivations(model).eval()

    logger.info("Saving normalization stats...")
    with open(output_dir / "normalization_stats.json", "w", encoding="utf-8") as f:
        json.dump({"vib_mean": vib_mean.tolist(), "vib_std": vib_std.tolist()}, f, indent=2)

    logger.info("Preparing a real test.csv row for export tracing, parity check, and benchmarking...")
    sample_row = test_df.iloc[[0]]
    sample_image = load_images_for_df(sample_row, project_root)
    sample_raw_vib = extract_raw_vib_features_from_df(sample_row, window_size, fs)
    sample_norm_vib = torch.tensor(
        apply_vibration_normalization(sample_raw_vib, vib_mean, vib_std), dtype=torch.float32,
    )

    logger.info("Exporting to ONNX...")
    onnx_path = output_dir / "fusion_model.onnx"
    export_model_to_onnx(wrapped_model, sample_image, sample_norm_vib, onnx_path)
    logger.info("Exported to %s", onnx_path)

    logger.info("Running parity check (PyTorch probability outputs vs. ONNX probability outputs)...")
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    parity_result = run_parity_check(wrapped_model, ort_session, sample_image, sample_norm_vib)
    logger.info("Parity check: defect_proba matches=%s, fault_proba matches=%s",
                parity_result["defect_matches"], parity_result["fault_matches"])
    if not parity_result["passed"]:
        raise RuntimeError(
            "ONNX export failed the parity check — exported model's probability outputs do not "
            "match the original PyTorch model's. Do not deploy this artifact."
        )

    logger.info("Benchmarking PyTorch vs. ONNX Runtime latency (n_runs=%d, batch=1, CPU)...", N_BENCHMARK_RUNS)
    torch_runner = make_torch_runner(wrapped_model, sample_image, sample_norm_vib)
    onnx_runner = make_onnx_runner(ort_session, sample_image, sample_norm_vib)
    torch_timing = benchmark_latency(torch_runner, N_BENCHMARK_RUNS, N_WARMUP_RUNS)
    onnx_timing = benchmark_latency(onnx_runner, N_BENCHMARK_RUNS, N_WARMUP_RUNS)

    logger.info("PyTorch:      mean=%.2fms (std=%.2fms)", torch_timing["mean_ms"], torch_timing["std_ms"])
    logger.info("ONNX Runtime: mean=%.2fms (std=%.2fms)", onnx_timing["mean_ms"], onnx_timing["std_ms"])
    speedup = torch_timing["mean_ms"] / onnx_timing["mean_ms"]
    logger.info("ONNX Runtime is %.2fx the speed of PyTorch on this single-sample, FP32, CPU benchmark.", speedup)

    logger.info("Export complete. Artifacts saved to %s", output_dir)