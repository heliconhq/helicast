import logging
import re
from typing import List, Literal, Optional, Sequence, Union

import holidays
import numpy as np
import pandas as pd

from helicast.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

__all__ = [
    "add_lagged_columns",
    "add_future_columns",
    "TestClass",
    "add_day_of_week_column",
    "add_day_of_year_column",
    "add_hour_column",
    "select_columns_by_regex",
    "remove_columns_by_regex",
    "add_sweden_public_holiday_column",
]


def _add_shifted_columns(
    df: pd.DataFrame,
    shifts: Sequence[int],
    columns: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    ignore_timestamps: bool = True,
    dropna: bool = False,
) -> pd.DataFrame:

    if exclude is None:
        exclude = []

    if columns is None:
        columns = df.columns.to_list()

    if ignore_timestamps:
        timestamps_columns = df.select_dtypes(include="datetime").columns.to_list()
        logger.info(f"Ignoring timestamps columns {timestamps_columns}")
        exclude.extend(timestamps_columns)

    shifted_columns = list(set(columns) - set(exclude))

    added_columns = []
    for c in shifted_columns:
        for i in shifts:
            if i >= 0:
                new_column_name = f"{c}_lag_{i}"
            else:
                new_column_name = f"{c}_future_{-i}"
            added_columns.append(new_column_name)
            df[new_column_name] = df[c].shift(i)

    logger.info(f"Added columns {added_columns}")

    if dropna:
        df = df.dropna()

    return df


def add_lagged_columns(
    df: pd.DataFrame,
    n: Union[int, Sequence[int]],
    columns: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    ignore_timestamps: bool = True,
    dropna: bool = False,
) -> pd.DataFrame:
    """Add lagged columns to the DataFrame ``df``. The lagged columns will have the
    same label as their original column with the ``_lagged_1``, ``_lagged_2``, ...
    appended to them.

    .. warning::

        Internally, the lag operator is implemented using ``pd.Series.shift(...)``,
        which supposes that the DataFrame ``df`` is a properly ordered time series
        DataFrame!

    Args:
        df: Input pandas DataFrame.
        n: If ``n`` is an integer, the lags will range from ``1`` to ``n+1`` included
         (``range(1, n + 1)``). If ``n`` is a sequence of integers
         (e.g., ``n = [1, 3, 5]``), only those specific values are considered. **Only
         positive values are allowed!**
        columns: List of columns to use. If ``None``, all columns are possibly used
         (depending on ``exclude`` and ``ignore_timestamps``). Defaults to None.
        exclude: List of columns to exclude. This paramater takes precendence over
         ``columns``. Defaults to None.
        ignore_timestamps: If ``True``, columns which have a datetime type are excluded.
         This paramater takes precendence over ``columns``. Defaults to True.
        dropna: If ``True``, rows containing NaNs are removed. Defaults to False.

    Returns:
        A DataFrame with additional columns containing lagged columns.
    """

    if isinstance(n, int):
        n = list(range(1, n + 1))

    for i in n:
        if i < 0:
            raise ValueError(f"n must specify positive values. Found {i}")

    df = _add_shifted_columns(
        df,
        shifts=n,
        columns=columns,
        exclude=exclude,
        ignore_timestamps=ignore_timestamps,
        dropna=dropna,
    )

    return df


def add_future_columns(
    df: pd.DataFrame,
    n: Union[int, Sequence[int]],
    columns: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    ignore_timestamps: bool = True,
    dropna: bool = False,
) -> pd.DataFrame:
    """Add future columns to the DataFrame ``df``. The future columns will have the
    same label as their original column with the ``_future_1``, ``_future_2``, ...
    appended to them.

    .. warning::

        Internally, the future operator is implemented using ``pd.Series.shift(...)``,
        which supposes that the DataFrame ``df`` is a properly ordered time series
        DataFrame!

    Args:
        df: Input pandas DataFrame.
        n: If ``n`` is an integer, the future will range from ``1`` to ``n+1`` included
         (``range(1, n + 1)``). If ``n`` is a sequence of integers
         (e.g., ``n = [1, 3, 5]``), only those specific values are considered. **Only
         positive values are allowed!**
        columns: List of columns to use. If ``None``, all columns are possibly used
         (depending on ``exclude`` and ``ignore_timestamps``). Defaults to None.
        exclude: List of columns to exclude. This paramater takes precendence over
         ``columns``. Defaults to None.
        ignore_timestamps: If ``True``, columns which have a datetime type are excluded.
         This paramater takes precendence over ``columns``. Defaults to True.
        dropna: If ``True``, rows containing NaNs are removed. Defaults to False.

    Returns:
        A DataFrame with additional columns containing future columns.
    """

    if isinstance(n, int):
        n = list(range(1, n + 1))

    for i in n:
        if i < 0:
            raise ValueError(f"n must specify positive values. Found {i}")

    # Negative shifts create future shifts!
    n = [-i for i in n]

    df = _add_shifted_columns(
        df,
        shifts=n,
        columns=columns,
        exclude=exclude,
        ignore_timestamps=ignore_timestamps,
        dropna=dropna,
    )

    return df


def add_day_of_week_column(
    df: pd.DataFrame, timestamp_column: Optional[str] = None
) -> pd.DataFrame:
    """Add a column ``"day_of_week"`` to the DataFrame ``df``. The values are of type
    ``int64`` and range from 0 (Monday) to 6 (Sunday).

    Args:
        df: Input pandas DataFrame.
        timestamp_column: Column containing the timestamps. If ``None``, the DataFrame
         index is used instead. Defaults to None.

    Returns:
        The DataFrame ``df`` with the column ``"day_of_week"`` added.
    """
    if timestamp_column is None:
        df["day_of_week"] = df.index.day_of_week.astype(int)
    else:
        df["day_of_week"] = df[timestamp_column].dt.day_of_week.astype(int)

    logging.info("Added column 'day_of_week' to the DataFrame.")

    return df


def add_day_of_year_column(
    df: pd.DataFrame, timestamp_column: Optional[str] = None
) -> pd.DataFrame:
    """Add a column ``"day_of_year"`` to the DataFrame ``df``. The values are of type
    ``int64``.

    Args:
        df: Input pandas DataFrame.
        timestamp_column: Column containing the timestamps. If ``None``, the DataFrame
         index is used instead. Defaults to None.

    Returns:
        The DataFrame ``df`` with the column ``"day_of_year"`` added.
    """
    if timestamp_column is None:
        df["day_of_year"] = df.index.day_of_year.astype(int)
    else:
        df["day_of_year"] = df[timestamp_column].dt.day_of_year.astype(int)

    logging.info("Added column 'day_of_year' to the DataFrame.")

    return df


def add_hour_column(
    df: pd.DataFrame, timestamp_column: Optional[str] = None
) -> pd.DataFrame:
    """Add a column ``"hour"`` to the DataFrame ``df``. The values are of type
    ``int64``.

    Args:
        df: Input pandas DataFrame.
        timestamp_column: Column containing the timestamps. If ``None``, the DataFrame
         index is used instead. Defaults to None.

    Returns:
        The DataFrame ``df`` with the column ``"hour"`` added.
    """
    if timestamp_column is None:
        df["hour"] = df.index.hour.astype(int)
    else:
        df["hour"] = df[timestamp_column].dt.hour.astype(int)

    logging.info("Added column 'hour' to the DataFrame.")

    return df


def remove_columns_by_regex(
    df: pd.DataFrame, regex: Union[str, List[str]]
) -> pd.DataFrame:
    """Remove columns by using regex.

    Args:
        df: Input pandas DataFrame.
        regex: Regex whose matches will remove columns. If ``regex`` is a list of
         strings, columns which match with at least one regex will be removed.

    Returns:
        A DataFrame whose columns matched ``regex`` are removed.
    """
    if isinstance(regex, str):
        regex = [regex]

    columns = set()
    for pattern in regex:
        matched_columns = [c for c in df.columns if re.search(pattern, c)]
        logger.info(
            f"Pattern r'{pattern}' matched with columns {sorted(matched_columns)}"
        )

        columns = columns.union(set(matched_columns))

    df = df.drop(columns=columns)

    logger.info(f"Removed columns {sorted(columns)}")

    return df


def select_columns_by_regex(
    df: pd.DataFrame, regex: Union[str, List[str]]
) -> pd.DataFrame:
    """Select columns by using regex.

    Args:
        df: Input pandas DataFrame.
        regex: Regex whose matches will select columns. If ``regex`` is a list of
         strings, columns which match with at least one regex will be selected.

    Returns:
        A DataFrame whose columns matched ``regex`` are the only one remaining.
    """

    if isinstance(regex, str):
        regex = [regex]

    columns = set()
    for pattern in regex:
        matched_columns = [c for c in df.columns if re.search(pattern, c)]
        logger.info(
            f"Pattern r'{pattern}' matched with columns {sorted(matched_columns)}"
        )

        columns = columns.union(set(matched_columns))

    drop_columns = set(df.columns) - columns

    df = df.drop(columns=drop_columns)

    logger.info(f"Selected columns {sorted(columns)}")

    return df


def _create_sweden_public_holiday_map(
    min_date: pd.Timestamp, max_date: pd.Timestamp, return_type="series"
):

    hd = pd.Series(holidays.SE(years=range(min_date.year, max_date.year + 1)))
    hd = hd[hd.str.lower() != "sunday"]
    hd.index = pd.to_datetime(hd.index).strftime("%Y-%m-%d")

    full_range = pd.date_range(min_date, max_date, freq="D").strftime("%Y-%m-%d")
    hd = hd.reindex(full_range)

    match return_type:
        case "series":
            pass
        case "dict":
            hd = hd.to_dict()
        case _:
            raise ValueError(
                f"`return_type` should be 'series' or 'dict'. Found: {return_type=}"
            )

    return hd


def add_sweden_public_holiday_column(
    df: pd.DataFrame,
    timestamp_column: Optional[str] = None,
    mode: Literal["int", "bool", "str"] = "int",
) -> pd.DataFrame:
    """Add a ``'sweden_public_holiday'`` column to the DataFrame ``df``. The holidays
    are:
    ``["All Saints' Day", 'Ascension Day', 'Christmas Day', 'Christmas Eve',
    'Easter Monday', 'Easter Sunday', 'Epiphany', 'Good Friday', 'May Day',
    'Midsummer Day', 'Midsummer Eve', 'National Day of Sweden', "New Year's Day",
    "New Year's Eve", 'Second Day of Christmas', 'Sunday; Whit Sunday']``

    If the public holiday falls on a Sunday, ``'; Sunday'`` will be appended to the
    public holiday name, e.g., ``'Christmas Day'`` will become
    ``'Christmas Day; Sunday'``.

    Args:
        df: Input DataFrame.
        timestamp_column: Column containing the timestamps. If ``None``, the DataFrame
         index is used instead. Defaults to None.
        mode: Controls the output type of the ``'sweden_public_holiday'`` column. If ``'int'``,
         it will fill the column with 1 and 0 (similarly for ``'bool'`` with ``True`` and
         ``False``). If ``'str'``, it will keep the names of the holidays and normal
         days will contains ``pd.NA``. Defaults to "int".

    Returns:
        The input DataFrame with an extra column ``'sweden_public_holiday'``
    """

    if timestamp_column is None:
        time_index = df.index.values
    else:
        time_index = df[timestamp_column]

    hd = _create_sweden_public_holiday_map(time_index.min(), time_index.max())

    df["sweden_public_holiday"] = hd[time_index.dt.strftime("%Y-%m-%d")].values

    match mode:
        case "int":
            df["sweden_public_holiday"] = (
                df["sweden_public_holiday"].notna().astype(int)
            )
        case "bool":
            df["sweden_public_holiday"] = df["sweden_public_holiday"].notna()
        case "str":
            pass
        case _:
            raise ValueError()

    return df


class TestClass:
    def method(self, i: int):
        """Dummy method

        Args:
            i: _description_
        """
        return None
