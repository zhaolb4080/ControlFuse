from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return config
