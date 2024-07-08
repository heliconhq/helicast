from datetime import timedelta
from typing import Annotated, Union

import pandas as pd
from pydantic.functional_validators import BeforeValidator

__all__ = [
    # Annotated types
    "Timedelta",
    "PositiveTimedelta",
    "NegativeTimedelta",
    "NonNegativeTimedelta",
    "NonPositiveTimedelta",
]


def _validate_timedelta(value: Union[pd.Timedelta, timedelta, str]) -> pd.Timedelta:
    """Validates the input value as a timedelta. The value can be a string, a pd.Timedelta
    or a datetime.timedelta object. Raises a ValueError if the conversion did not work."""
    if isinstance(value, pd.Timedelta):
        return value
    elif isinstance(value, (str, timedelta)):
        try:
            return pd.Timedelta(value)
        except Exception:
            pass
    raise ValueError(f"Couldn't convert {value=} ({type(value)=}) to a pd.Timedelta.")


def _validate_positive_timedelta(
    value: Union[pd.Timedelta, timedelta, str],
) -> pd.Timedelta:
    """Validates the input value as a positive timedelta. Note that a time delta of 0
    is not valid."""
    value = _validate_timedelta(value)
    if value <= pd.Timedelta(0):
        raise ValueError(f"Timedelta should be positive. Found {value=}.")
    return value


def _validate_negative_timedelta(
    value: Union[pd.Timedelta, timedelta, str],
) -> pd.Timedelta:
    """Validates the input value as a negative timedelta. Note that a time delta of 0 is
    not valid."""
    value = _validate_timedelta(value)
    if value >= pd.Timedelta(0):
        raise ValueError(f"Timedelta should be negative. Found {value=}.")
    return value


def _validate_non_negative_timedelta(
    value: Union[pd.Timedelta, timedelta, str],
) -> pd.Timedelta:
    """Validates the input value as a non-negative timedelta. Note that a time delta of
    0 is a proper non-negative timedelta."""
    value = _validate_timedelta(value)
    if value < pd.Timedelta(0):
        raise ValueError(f"Timedelta should be non-negative. Found {value=}.")
    return value


def _validate_non_positive_timedelta(
    value: Union[pd.Timedelta, timedelta, str],
) -> pd.Timedelta:
    """Validates the input value as a non-positive timedelta. Note that a time delta of
    0 is a proper non-positive timedelta."""
    value = _validate_timedelta(value)
    if value > pd.Timedelta(0):
        raise ValueError(f"Timedelta should be non-positive. Found {value=}.")
    return value


Timedelta = Annotated[pd.Timedelta, BeforeValidator(_validate_timedelta)]
#: Annotated type for timedelta. Can be used with Pydantic.


#: Annotated type for positive timedelta. Can be used with Pydantic.
PositiveTimedelta = Annotated[
    pd.Timedelta, BeforeValidator(_validate_positive_timedelta)
]

#: Annotated type for negative timedelta. Can be used with Pydantic.
NegativeTimedelta = Annotated[
    pd.Timedelta, BeforeValidator(_validate_negative_timedelta)
]

#: Annotated type for non-negative timedelta. Can be used with Pydantic.
NonNegativeTimedelta = Annotated[
    pd.Timedelta, BeforeValidator(_validate_non_negative_timedelta)
]

#: Annotated type for non-positive timedelta. Can be used with Pydantic.
NonPositiveTimedelta = Annotated[
    pd.Timedelta, BeforeValidator(_validate_non_positive_timedelta)
]
