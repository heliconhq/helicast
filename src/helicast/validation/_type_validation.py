from typing import Any, Generic, List, Type, TypeVar

from pydantic import TypeAdapter, ValidationError

__all__ = [
    "ScalarToListValidator",
    "validate_scalar_to_list",
]

T = TypeVar("T")


class ScalarToListValidator(Generic[T]):
    def __init__(self, _type: Type[T]):
        if not isinstance(_type, type):
            raise TypeError(f"{_type=} must be of type 'type'.")
        self.type = _type

    def __call__(self, value: Any) -> List[T]:
        try:
            config = {"arbitrary_types_allowed": True}
            value = TypeAdapter(self.type, config=config).validate_python(value)
            value = [value]
        except ValidationError:
            pass
        return TypeAdapter(List[self.type], config=config).validate_python(value)


def validate_scalar_to_list(value: Any, _type: Type[T]) -> List[T]:
    """Validate a scalar value to a list of the specified type.

    .. code-block:: python

        from helicast.validation import validate_scalar_to_list

        print(validate_scalar_to_list(1, int))
        # [1]
        print(validate_scalar_to_list([1], int))
        # [1]
        print(validate_scalar_to_list([1], list[int]))
        # [[1]]
        print(validate_scalar_to_list(1, list[int]))
        # ValidationError


    Args:
        value: Scalar value to be validated to a list. If the value is already a list
            of scalars defined by ``_type``, it is returned as is. If the value is a
            scalar defined by ``_type``, it is returned as a list of one element.
        _type: Type of the scalar value within the list.

    Returns:
        List of scalar values of type ``_type``.
    """
    return ScalarToListValidator(_type)(value)
