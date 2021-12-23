# yaml-models

Python package generating `torch` models from a `yaml` configuration file.
Each `type` item in model list need to specify at minimum the default values
needed by the `torch.nn` module. Arguments not specified will use the default
values.

Example config:

```yaml
model:
-   type: Linear
    in_features: 2
    out_features: 10
-   type: ReLU
-   type: Linear
    in_features: 10
    out_features: 1
-   type: Sigmoid
```

`torch.nn.Linear` needs the default arguments `in_features` and `out_features`.

## Usage

```consol
pip install yaml-models
```

```pycon
>>> from yaml_models.model import Model
>>> model = Model(config_path="./example_config/model.yaml")
>>> print(model)
Model(
  (layers): ModuleList(
    (0): Linear(in_features=2, out_features=10, bias=True)
    (1): ReLU()
    (2): Linear(in_features=10, out_features=1, bias=True)
    (3): Sigmoid()
  )
)
```
