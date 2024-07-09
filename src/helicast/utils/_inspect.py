from dataclasses import fields
from typing import Any, ClassVar, Dict, List

__all__ = [
    "is_classvar",
    "get_param_type_mapping",
    "get_classvar_list",
]


def is_classvar(_type: Any) -> bool:
    """Returns True if the ``_type`` is a ClassVar, False otherwise."""
    if _type == ClassVar:
        return True

    try:
        if _type.__origin__ == ClassVar:
            return True
    except Exception:
        pass

    return False


def get_param_type_mapping(cls) -> Dict[str, type]:
    """Returns the types of the parameters of the estimator. This is useful for
    type checking. Correctly ignores ClassVar fields."""

    return {p.name: p.type for p in fields(cls)}


def get_classvar_list(cls) -> List[str]:
    """Returns the list of ClassVar fields of the dataclass."""
    __annotations__ = getattr(cls, "__annotations__", {})
    return [name for name, _type in __annotations__.items() if is_classvar(_type)]
