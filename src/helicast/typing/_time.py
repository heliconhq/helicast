from datetime import date, datetime, timedelta
from typing import Annotated, Union

import pandas as pd
from pydantic.functional_validators import AfterValidator, BeforeValidator

__all__ = [
    # Annotated types for TIMESTAMP
    "Timestamp",
    "TzAwareTimestamp",
    "TzNaiveTimestamp",
    # Annotated types for TIMEDELTA
    "Timedelta",
    "PositiveTimedelta",
    "NegativeTimedelta",
    "NonNegativeTimedelta",
    "NonPositiveTimedelta",
]


#################
### TIMESTAMP ###
#################


def _validate_timestamp(
    value: Union[pd.Timestamp, datetime, date, str],
) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    elif isinstance(value, (datetime, date, str)):
        return pd.Timestamp(value)
    else:
        ValueError(
            f"Unsupported type for {value=} ({type(value)=}). "
            f"Expected pd.Timestamp, datetime, date or str."
        )


def _validate_tz_aware(value: pd.Timestamp) -> pd.Timestamp:
    if value.tz is None:
        raise ValueError(f"Timestamp must be timezone aware. Found {value=}.")
    return value


def _validate_tz_naive(value: pd.Timestamp) -> pd.Timestamp:
    if value.tz is not None:
        raise ValueError(f"Timestamp must be timezone naive. Found {value=}.")
    return value


#: Annotated type for timestamp. Can be used with Pydantic.
Timestamp = Annotated[pd.Timestamp, BeforeValidator(_validate_timestamp)]

#: Annotated type for time-zone aware timestamps. Can be used with Pydantic.
TzAwareTimestamp = Annotated[Timestamp, AfterValidator(_validate_tz_aware)]

#: Annotated type for time-zone aware timestamps. Can be used with Pydantic.
TzNaiveTimestamp = Annotated[Timestamp, AfterValidator(_validate_tz_naive)]


#################
### TIMEDELTA ###
#################


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
