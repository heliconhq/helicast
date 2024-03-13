import logging
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from helicast.logging import configure_logging

configure_logging()

logger = logging.getLogger(__name__)

__all__ = ["add_lagged_columns", "add_future_columns", "TestClass"]


def _add_shifted_columns(
    df: pd.DataFrame,
    shifts: Sequence[int],
    columns: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    ignore_timestamps: bool = True,
    dropna: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

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
    inplace: bool = False,
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
        inplace: If ``True``, works with the DataFrame object in place, otherwise
         works with a copy. Defaults to False.

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
        inplace=inplace,
    )

    return df


def add_future_columns(
    df: pd.DataFrame,
    n: Union[int, Sequence[int]],
    columns: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    ignore_timestamps: bool = True,
    dropna: bool = False,
    inplace: bool = False,
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
        inplace: If ``True``, works with the DataFrame object in place, otherwise
         works with a copy. Defaults to False.

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
        inplace=inplace,
    )

    return df


class TestClass:
    def method(self, i: int):
        """Dummy method

        Args:
            i: _description_
        """
        return None
