import pytest

from yaml_models.config import _load_config


def test_load_config(config_path):
    res = _load_config(config_path)

    assert len(res) == 4

    assert res[0][0] == "Linear"
    assert all([
        key in ["in_features", "out_features"]
        for key in res[0][1].keys()
    ])


def test_load_no_config(tmpdir):
    path = tmpdir.join("model.yaml")

    with pytest.raises(FileNotFoundError):
        _ = _load_config(path)
