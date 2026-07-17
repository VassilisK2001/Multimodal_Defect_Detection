
from pathlib import Path
import yaml


def find_project_root(marker: str = "pyproject.toml") -> Path:
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root (looked for {marker})")

def load_yaml_config(relative_path: str) -> dict:
    """Load a YAML config file, with the path resolved relative to the project root"""
    project_root = find_project_root()
    with open(project_root / relative_path) as f:
        return yaml.safe_load(f)