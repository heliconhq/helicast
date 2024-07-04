from logging import getLogger
from typing import List, Union

import numpy as np
import pandas as pd

from helicast.logging import configure_logging

__all__ = [
    "auto_convert_to_datetime_index",
]

configure_logging()
logger = getLogger(__name__)

DATETIME_FORMATS = [
    "%Y/%m/%d %H:%M:%S",
    "%Y/%d/%m %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
]
for i in range(len(DATETIME_FORMATS)):
    date_format = DATETIME_FORMATS[i]
    DATETIME_FORMATS.append(date_format.replace("/", "-"))
for i in range(len(DATETIME_FORMATS)):
    date_format = DATETIME_FORMATS[i]
    DATETIME_FORMATS.append(date_format.replace("Y", "y"))

DATE_FORMATS = [i.split(" ")[0] for i in DATETIME_FORMATS]


def auto_convert_to_datetime_index(
    dates: Union[np.ndarray, List, pd.Series, pd.DatetimeIndex],
) -> pd.DatetimeIndex:
    """Auto-convert an array-like object of dates (as string) to a pd.DatetimeIndex.

    Args:
        dates: Array-like object of dates to convert to pd.DatetimeIndex.

    Raises:
        TypeError: If the input is not a np.ndarray, a list, a pd.Series or
            pd.DatetimeIndex.
        RuntimeError: If no format is found.
        RuntimeError: If several formats with the same number of frequencies are found.

    Returns:
        The dates converted to a pd.DatetimeIndex.
    """

    if isinstance(dates, np.ndarray):
        dates = pd.Series(dates)
    elif isinstance(dates, list):
        dates = pd.Series(dates)

    if isinstance(dates, pd.DatetimeIndex):
        logger.warning("Input is already pd.DatetimeIndex, returning it as it is!")
        return dates
    elif isinstance(dates, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(dates):
            logger.warning(
                "Input is already a timeseries pd.Series, returning it as a pd.DatetimeIndex!"
            )
            return pd.DatetimeIndex(dates)
    else:
        raise TypeError(
            f"Input should be a pd.Series (or pd.DatetimeIndex), got {type(dates)}!"
        )

    results = {}
    for format in DATETIME_FORMATS + DATE_FORMATS:
        try:
            converted = pd.to_datetime(dates, format=format)
            results[format] = converted
        except:
            continue

    if len(results) == 0:
        raise RuntimeError("No format found!")
    elif len(results) == 1:
        return pd.DatetimeIndex(list(results.values())[0])

    results_frequencies = {}
    for k in list(results.keys()):
        results_frequencies[k] = results[k].diff().value_counts()

    # Create a list of tuple of (n_frequencies, frequencies, format)
    formats = [(len(v), v, k) for k, v in results_frequencies.items()]
    formats = sorted(formats, key=lambda x: x[0])
    if formats[0][0] == formats[1][0]:
        raise RuntimeError(f"Multiple formats found, {results}")
    return pd.DatetimeIndex(results[formats[0][2]])
