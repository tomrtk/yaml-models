from __future__ import annotations

import torch
from torch import nn

from yaml_models.config import _load_config


class MissingResquiredArgument(Exception):
    pass


class Model(nn.Module):
    def __init__(self, config_path: str) -> None:
        super().__init__()

        layer_def = _load_config(path=config_path)

        layers = []
        for k, v in layer_def:
            cls = getattr(nn, k)
            try:
                layer = cls(**v)
            except TypeError:
                raise MissingResquiredArgument(
                    f"missing default arguments for {k}",
                )
            else:
                layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return y
