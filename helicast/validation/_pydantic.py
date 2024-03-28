from pydantic import BaseModel

__all__ = ["HelicastBaseModel"]


class HelicastBaseModel(BaseModel):
    """Alternative Pydantic Basemodel that enforces the following config to
    child classes:

    .. code:: python

        _enforced_config = {
            "arbitrary_types_allowed": True,
            "validate_assignment": True,
            "extra": "forbid",
        }

    """

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        _enforced_config = {
            "arbitrary_types_allowed": True,
            "validate_assignment": True,
            "extra": "forbid",
        }

        cls.model_config.update(_enforced_config)
