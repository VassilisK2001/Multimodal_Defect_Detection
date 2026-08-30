"""
Splits manifest.csv into stratified train/val/test CSVs.
"""

import logging

import pandas as pd

from defect_detection.data.splitting import split_manifest
from defect_detection.utils import find_project_root, load_yaml_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    config = load_yaml_config("config/data_config.yaml")
    project_root = find_project_root()

    manifest_path = project_root / config["paths"]["manifest_dir"] / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)

    result_df = split_manifest(manifest_df)

    out_dir = project_root / config["paths"]["manifest_dir"]
    for split_name in ["train", "val", "test"]:
        subset = result_df[result_df.split == split_name].drop(columns=["split"])
        out_path = out_dir / f"{split_name}.csv"
        subset.to_csv(out_path, index=False)
        logger.info("%s: %d rows -> %s", split_name, len(subset), out_path)
        logger.info("  is_defect: %s", subset["is_defect"].value_counts().to_dict())
        logger.info("  fault_class: %s", subset["fault_class"].value_counts().to_dict())