import pytest

CONFIG = """\
model:
-   type: Linear
    in_features: 2
    out_features: 10
-   type: ReLU
-   type: Linear
    in_features: 10
    out_features: 1
-   type: Sigmoid
"""

CONFIG_MISSING_FIELD = """\
model:
-   type: Linear
    in_features: 2
"""

CONFIG_BAD_YAML = """\
model:
  type: Linear
    in_features: 2
"""


@pytest.fixture
def config_path(tmpdir) -> str:
    config_path = tmpdir.join("model.yaml")

    with open(config_path, "w") as f:
        f.write(CONFIG)

    return config_path


@pytest.fixture
def config_missing_fields(tmpdir) -> str:
    config_path = tmpdir.join("model.yaml")

    with open(config_path, "w") as f:
        f.write(CONFIG_MISSING_FIELD)

    return config_path


@pytest.fixture
def config_bad_yaml(tmpdir) -> str:
    config_path = tmpdir.join("model.yaml")

    with open(config_path, "w") as f:
        f.write(CONFIG_BAD_YAML)

    return config_path
