import logging

from defect_detection.data.manifest import build_manifest
from defect_detection.utils import find_project_root, load_yaml_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    config = load_yaml_config("config/data_config.yaml")
    df = build_manifest()

    project_root = find_project_root()
    out_dir = project_root / config["paths"]["manifest_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest.csv"
    df.to_csv(out_path, index=False)

    logger.info("Manifest written to %s (%d rows)", out_path, len(df))
    logger.info("is_defect counts:\n%s", df["is_defect"].value_counts())
    logger.info("fault_class counts:\n%s", df["fault_class"].value_counts())