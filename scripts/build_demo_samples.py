""""
Selects test.csv rows and extracts them as preloaded demo samples
for the deployed app. It extracts one image and one raw vibration 
window per sample. Preferred rows are computed directly from the 
deployed model's real test-set predictions, prioritizing 
correctly-classified examples.
"""

import json
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.export.demo_samples import build_manifest_entry, select_demo_rows, write_demo_sample
from defect_detection.interpretability.example_selection import find_correct_normal_examples, find_fault_type_examples
from defect_detection.mlflow_utils import load_model_and_stats
from defect_detection.utils import find_project_root, load_yaml_config


logger = logging.getLogger(__name__)

N_PER_SCENARIO = 3


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")
    test_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "test.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    window_size, fs = data_config["window_size"], data_config["cwru"]["sampling_rate_hz"]
    output_root = project_root / "deployment" / "demo_samples"
    device = torch.device("cpu")

    logger.info("Loading final deployed model and computing test-set predictions...")
    model, vib_mean, vib_std = load_model_and_stats("both", device=device)
    test_dataset = MultimodalDefectDataset(
        test_df, window_size=window_size, fs=fs, training=False, vib_mean=vib_mean, vib_std=vib_std)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    predictions = collect_test_predictions(model, test_loader, device=device)

    logger.info("Identifying correctly-classified rows to prefer for demo samples...")
    preferred_normal_indices = find_correct_normal_examples(predictions)

    fault_examples = find_fault_type_examples(predictions, class_names, n=N_PER_SCENARIO)
    preferred_fault_indices = [
        entry["row_index"]
        for class_name in class_names
        for entry in fault_examples[class_name]["correct"]
    ]

    all_preferred_indices = preferred_normal_indices + preferred_fault_indices

    logger.info("Selecting %d demo rows per scenario...", N_PER_SCENARIO)
    scenarios = select_demo_rows(
        test_df, class_names, n_per_class=N_PER_SCENARIO, preferred_row_indices=all_preferred_indices,
    )

    manifest_entries = []
    for scenario_name, rows in scenarios.items():
        logger.info("Writing %d sample(s) for scenario '%s'...", len(rows), scenario_name)
        for i, (_, row) in enumerate(rows.iterrows(), start=1):
            sample_name = f"sample_{i}"
            output_dir = output_root / scenario_name / sample_name
            write_demo_sample(row, project_root, window_size, output_dir)
            manifest_entries.append(build_manifest_entry(scenario_name, sample_name, row))

    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    logger.info("Demo samples complete. %d total samples written to %s", len(manifest_entries), output_root)