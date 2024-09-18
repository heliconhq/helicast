from dataclasses import fields
from functools import partial, wraps
from typing import Any, Callable, ClassVar, Dict, List

__all__ = [
    "is_classvar",
    "get_param_type_mapping",
    "get_classvar_list",
    "has_method",
    "is_fitable",
    "is_transformer",
    "is_invertible_transformer",
    "is_predictor",
    "set_method_flags",
    "check_method",
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


def has_method(obj: type | object, method_name: str) -> bool:
    """Return True if the instance or class has a method with the given name, False
    otherwise."""
    return callable(getattr(obj, method_name, None))


def _is_capable_of(obj: type | object, method_name: str) -> bool:
    """Check if an instance or a class has a method with the given name. If the instance
    or class has an attribute '_can_{method_name}' (e.g., '_can_transform'),, it will be
    used to determine if the instance or class is capable of performing the method."""
    if hasattr(obj, f"_can_{method_name}"):
        return getattr(obj, f"_can_{method_name}")
    return has_method(obj, method_name)


def is_fitable(obj: type | object) -> bool:
    """Check if an instance or a class is fitable:

    * If ``obj`` has an attribute ``_can_fit``, return the value of this attribute.
    * Else, check if the instance or class has a method ``fit``."""
    return _is_capable_of(obj, "fit")


def is_transformer(obj: type | object) -> bool:
    """Check if an instance or a class is a transformer, by first checking if the
    instance or class is fitable, and, if yes, checking:

    * If ``obj`` has an attribute ``_can_transform``, return the value of this attribute.
    * Else, check if the instance or class has a method ``transform``."""
    return is_fitable(obj) and _is_capable_of(obj, "transform")


def is_invertible_transformer(obj: type | object) -> bool:
    """Check if an instance or a class is an invertible transforme, by first checking if
    the instance or class is a transformer, and, if yes, checking:

    * If ``obj`` has an attribute ``_can_inverse_transform``, return the value of
    this attribute.
    * Else, check if the instance or class has a method ``inverse_transform``."""
    return is_transformer(obj) and _is_capable_of(obj, "inverse_transform")


def is_predictor(obj: type | object) -> bool:
    """Check if an instance or a class is a predictor, by first checking if the
    instance or class is fitable, and, if yes, checking:

    * If ``obj`` has an attribute ``_can_predict``, return the value of this attribute.
    * Else, check if the instance or class has a method ``predict``."""
    return is_fitable(obj) and _is_capable_of(obj, "predict")


def set_method_flags(obj: type | object):
    obj._can_fit = is_fitable(obj)
    obj._can_transform = is_transformer(obj)
    obj._can_inverse_transform = is_invertible_transformer(obj)
    obj._can_predict = is_predictor(obj)


def check_method(method=None, *, check: Callable):
    if method is None:
        return partial(check_method, check=check)

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not check(self):
            raise RuntimeError(f"check is not '{check.__name__}'.")
        return method(self, *args, **kwargs)

    return wrapper
