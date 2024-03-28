from typing import Any, List

from pydantic import AfterValidator, TypeAdapter, ValidationError

__all__ = [
    "ScalarToListValidator",
    "validate_scalar_to_list",
]


class ScalarToListValidator:
    def __init__(self, __type: type):
        if not isinstance(__type, type):
            raise TypeError(f"{__type=} must be of type 'type'.")
        self.type = __type

    def __call__(self, v: Any) -> List[any]:
        try:
            config = {"arbitrary_types_allowed": True}
            scalar = TypeAdapter(self.type, config=config).validate_python(scalar)
            scalar = [scalar]
        except ValidationError:
            pass
        return TypeAdapter(List[self.type], config=config).validate_python(scalar)


def validate_scalar_to_list(v: Any, __type: type) -> List[Any]:

    return ScalarToListValidator(__type)(v)
