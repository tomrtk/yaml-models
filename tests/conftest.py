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

BAD_CONFIG = """\
model:
-   type: Linear
    in_features: 2
"""


@pytest.fixture
def config_path(tmpdir) -> str:
    config_path = tmpdir.join("model.yaml")

    with open(config_path, "w") as f:
        f.write(CONFIG)

    return config_path


@pytest.fixture
def bad_config_path(tmpdir) -> str:
    config_path = tmpdir.join("model.yaml")

    with open(config_path, "w") as f:
        f.write(BAD_CONFIG)

    return config_path
