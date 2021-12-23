import pytest
import torch

from yaml_models.model import MissingResquiredArgument
from yaml_models.model import Model


def test_model(config_path):
    model = Model(config_path=config_path)

    print(model)
    assert len(model.layers) == 4

    x = torch.rand((1, 2))

    pred = model(x)

    assert pred.shape == (1, 1)


def test_load_bad_config(bad_config_path):
    with pytest.raises(MissingResquiredArgument):
        _ = Model(config_path=bad_config_path)
