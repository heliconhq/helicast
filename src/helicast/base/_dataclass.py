from functools import partial

import pydantic.dataclasses
from typing_extensions import dataclass_transform

__all__ = [
    "dataclass",
]

DEFAULT_CONFIG = {"arbitrary_types_allowed": True}


@dataclass_transform()
def dataclass(cls=None, *, config=None):
    """Dataclass decorator for the Helicast library. It uses Pydantic under the hood,
    with minimal checks to ensure that nothing breaks. The main difference with the
    Pydantic dataclass is that the default config is set to allow arbitrary types.
    Also, the 'validate_assignment' cannot be set to True in the config (otherwise it
    would break on-the-fly assignements).
    """
    if cls is None:
        return partial(dataclass, config=config)

    if config is not None and config.get("validate_assignment", False) is True:
        raise RuntimeError(
            "Do not use 'validate_assignment=True' in the Pydantic config in child classes!"
        )
    if config is None:
        config = DEFAULT_CONFIG

    return pydantic.dataclasses.dataclass(_cls=cls, config=config)
