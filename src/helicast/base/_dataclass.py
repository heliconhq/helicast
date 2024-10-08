from functools import partial

import pydantic.dataclasses
from typing_extensions import dataclass_transform

__all__ = [
    "dataclass",
]


_DEFAULT_CONFIG = {
    "arbitrary_types_allowed": True,
    "extra": "forbid",
    "validate_assignment": False,
}


def _validate_config(config: dict | None = None) -> dict:
    """Validate the config dict to be used in the dataclass decorator. See
    ``_DEFAULT_CONFIG`` for the defaults."""
    if config is None:
        config = {}

    for k, v in _DEFAULT_CONFIG.items():
        if k in config.keys() and config[k] != v:
            raise ValueError(
                f"Config parameter '{k}' expects value {repr(v)} but got {config[k]}."
            )
        config[k] = v
    return config


@dataclass_transform()
def dataclass(cls=None, *, config=None):
    """Dataclass decorator for the Helicast library. It uses Pydantic under the hood,
    with minimal checks to ensure that nothing breaks. The default config is set to

    * ``"arbitrary_types_allowed": True``
    * ``"extra": "forbid"``
    * ``""validate_assignment": False``

    Any other value for those specific keys will raise a ``ValueError``.
    """
    if cls is None:
        return partial(dataclass, config=config)

    config = _validate_config(config)

    return pydantic.dataclasses.dataclass(_cls=cls, config=config)
