import pytest
import torch

from yaml_models.model import MissingResquiredArgument
from yaml_models.model import Model


def test_model(config_path):
    model = Model(config_path=config_path)

    assert len(model.layers) == 4

    x = torch.rand((1, 2))

    pred = model(x)

    assert pred.shape == (1, 1)


def test_load_model_config_missing_fields(config_missing_fields):
    with pytest.raises(MissingResquiredArgument):
        _ = Model(config_path=config_missing_fields)
