from __future__ import annotations

import os.path

import yaml


class YAMLError(Exception):
    pass


def _load_config(path: str) -> list[tuple[str, dict[str, str | int]]]:
    if os.path.exists(path) is False:
        raise FileNotFoundError(f"config file {path} not found")

    try:
        config = yaml.safe_load(open(path))
    except yaml.scanner.ScannerError as e:
        raise YAMLError(f"error in config file, {e}")

    return [(val.pop("type"), val) for val in config["model"]]
