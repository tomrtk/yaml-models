from __future__ import annotations

import os.path

import yaml


def _load_config(path: str) -> list[tuple[str, dict[str, str | int]]]:
    if os.path.exists(path) is False:
        raise FileNotFoundError(f"config file {path} not found")

    config = yaml.safe_load(open(path))

    return [(val.pop("type"), val) for val in config["model"]]
