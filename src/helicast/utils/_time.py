from typing import Dict, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import validate_call
from pydantic.dataclasses import dataclass
from strenum import StrEnum
from timezonefinder import TimezoneFinder

from helicast.typing import TzAwareTimestamp

__all__ = [
    "get_timezone",
    "TimeConverter",
    "convert_timestamp_to_float_seconds",
    "convert_float_seconds_to_timestamp",
]


#: This is EPOCH time, a.k.a., pd.Timestamp("1970-01-01", tz="UTC")
EPOCH_TIMESTAMP: pd.Timestamp = pd.Timestamp.fromtimestamp(0, tz="UTC")


class TimeUnit(StrEnum):
    """Enum for time units. Those are the unambiguous time units that define time
    intervals. Other units like months or years are ambiguous and cannot be used!

    Note that StrEnum is a subclass of str.
    """

    NANOSECOND = "ns"
    MICROSECOND = "us"
    MILLISECOND = "ms"
    SECOND = "s"
    MINUTE = "m"
    HOUR = "h"
    DAY = "D"
    WEEK = "W"


#: Conversion factors for different time units. The base unit is nanoseconds.
TIME_UNIT_CONVERSION_MAPPING: Dict[str, int] = {
    TimeUnit.NANOSECOND: int(1),
    TimeUnit.MICROSECOND: int(1e3),
    TimeUnit.MILLISECOND: int(1e6),
    TimeUnit.SECOND: int(1e9),
    TimeUnit.MINUTE: int(1e9 * 60),
    TimeUnit.HOUR: int(1e9 * 3600),
    TimeUnit.DAY: int(1e9 * 3600 * 24),
    TimeUnit.WEEK: int(1e9 * 3600 * 24 * 7),
}


def get_timezone(
    lat: float, long: float, raise_if_not_found: bool = False
) -> Union[str, None]:
    """Finds the timezone for a given latitude and longitude.

    Args:
        lat: Latitude of the location.
        long: Longitude of the location.
        raise_if_not_found: If True, raises a ``ValueError`` if the timezone is not
            found, otherwise, allows the function to return None. Defaults to False.

    Raises:
        ValueError: If ``raise_if_not_found=True`` and the timezone is not found.

    Returns:
        The timezone for the location if found, None otherwise.
    """
    tz = TimezoneFinder().timezone_at(lat=float(lat), lng=float(long))
    if tz is None and raise_if_not_found:
        raise ValueError(f"Timezone not found for {lat, long}.")
    return tz


@validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
def validate_tz_aware(
    x: Union[pd.Timestamp, pd.DatetimeIndex],
) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """Validate that a timestamp or DatetimeIndex is timezone-aware.

    Args:
        x: Timestamp or DatetimeIndex to validate.

    Returns:
        The validated ``x`` object.
    """
    if x.tzinfo is None:
        raise ValueError(f"Timestamp is not TZ-aware! Found {x.tzinfo=}.")
    return x


@dataclass(config={"arbitrary_types_allowed": True})
class TimeConverter:
    """Class to convert timestamps to arrays and vice-versa.

    Args:
        unit: Time unit to use for the conversion. Can be one of the values in
            TimeUnit (which is a StrEnum). Defaults to TimeUnit.SECOND.
        reference: Reference time to use for the conversion. Must be a timezone-aware
            pd.Timestamp. Defaults to UNIX epoch, i.e.,
            ``pd.Timestamp("1970-01-01", tz="UTC")``.
    """

    unit: TimeUnit = TimeUnit.SECOND
    reference: TzAwareTimestamp = EPOCH_TIMESTAMP

    def __post_init__(self):
        self._conversion_factor = TIME_UNIT_CONVERSION_MAPPING[self.unit]

    def convert_timestamp_to_array(
        self, timestamp: Union[pd.Timestamp, pd.DatetimeIndex]
    ) -> Union[float, NDArray[np.float64]]:
        """Converts a pd.Timestamp or pd.DatetimeIndex to a float or np.ndarray of
        float64.

        Args:
            timestamp: Timestamp to convert.

        Returns:
            A float or np.ndarray of float64 representing the time in units of
            ``self.unit`` since the reference time ``self.reference``.
        """

        validate_tz_aware(timestamp)

        delta = timestamp - self.reference
        if isinstance(timestamp, pd.Timestamp):
            delta = delta.value / self._conversion_factor
        else:
            delta = delta.values.astype(int) / self._conversion_factor
        return delta

    def convert_array_to_timestamp(
        self, array: Union[int, float, NDArray], tz: Union[str, None] = None
    ) -> Union[pd.Timestamp, pd.DatetimeIndex]:
        """Converts an array of floats to a pd.DatetimeIndex (or to a pd.Timestamp if the
        array is a scalar). The conversion depends on the ``self.reference`` and
        ``self.unit`` attributes.

        Args:
            array: Array of floats representing the time axis.
            tz: Timezone to convert the timestamp to. If None, the timestamp will be the
                same as the reference time. Defaults to None.

        Returns:
            A pd.Timestamp or pd.DatetimeIndex object.
        """
        array = np.asarray(array, dtype=np.float64)
        delta = (array * self._conversion_factor).astype(int)
        timestamp = self.reference + pd.to_timedelta(delta, unit="ns")
        if tz is not None:
            timestamp = timestamp.tz_convert(tz)
        return timestamp


def convert_timestamp_to_float_seconds(
    timestamp: Union[pd.Timestamp, pd.DatetimeIndex],
    reference: Union[pd.Timestamp, None] = None,
) -> Union[float, NDArray]:
    """Converts a pd.Timestamp or pd.DatetimeIndex to a float or np.ndarray of float64
    representing the time in seconds since the reference time.

    Args:
        timestamp: Timestamp to convert.
        reference: Reference time to use for the conversion. Must be a timezone-aware
            pd.Timestamp. If None, the reference time is set to UNIX epoch. Defaults to
            None.

    Returns:
        A float or np.ndarray of float64 representing the time in seconds since the
        reference time.
    """
    converter = TimeConverter(unit=TimeUnit.SECOND, reference=reference)
    return converter.convert_timestamp_to_array(timestamp)


def convert_float_seconds_to_timestamp(
    array: Union[int, float, NDArray],
    reference: pd.Timestamp = EPOCH_TIMESTAMP,
    tz: Union[str, None] = None,
) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """Converts an array of floats to a pd.DatetimeIndex (or to a pd.Timestamp if the
    array is a scalar) representing the time in seconds since the reference time.

    Args:
        array: Array of floats representing the time axis.
        reference: Reference time to use for the conversion. Must be a timezone-aware
            pd.Timestamp. If None, the reference time is set to UNIX epoch. Defaults to
            None.
        tz: Timezone to convert the timestamp to. If None, the timestamp will be the
            same as the reference time. Defaults to None.

    Returns:
        A pd.Timestamp or pd.DatetimeIndex object.
    """
    converter = TimeConverter(unit=TimeUnit.SECOND, reference=reference)
    return converter.convert_array_to_timestamp(array, tz=tz)
