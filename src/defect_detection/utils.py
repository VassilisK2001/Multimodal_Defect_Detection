
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

def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary into a single-level dict with joined key paths.
 
    Args:
        d: A dictionary, possibly containing nested dictionaries.
        parent_key: Prefix prepended to all keys.
        sep: Separator used to join nested key parts.
 
    Returns:
        A flat dict.
    """
    items = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items