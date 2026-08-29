import logging

from defect_detection.data.ingestion import download_dataset, verify_structure
from defect_detection.utils import find_project_root

logger = logging.getLogger(__name__)


MVTEC_KAGGLE_SLUG = "ipythonx/mvtec-ad"      
CWRU_KAGGLE_SLUG = "astrollama/cwru-case-western-reserve-university-dataset" 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    project_root = find_project_root()
    raw_data_dir = project_root / "data" / "raw"
    mvtec_dir = raw_data_dir / "mvtec"
    cwru_dir = raw_data_dir / "cwru"

    raw_data_dir.mkdir(parents=True, exist_ok=True)

    download_dataset(MVTEC_KAGGLE_SLUG, mvtec_dir, "MVTec AD")
    download_dataset(CWRU_KAGGLE_SLUG, cwru_dir, "CWRU")

    verify_structure(mvtec_dir, cwru_dir)