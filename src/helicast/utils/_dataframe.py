from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import validate_call

from helicast.utils._collections import maybe_reorder_like

__all__ = [
    "series_to_dataframe",
    "validate_column_names_as_string",
    "numpy_array_to_dataframe",
    "maybe_reorder_columns",
    "find_datetime_index_unique_frequency",
    "find_common_indices",
    "restrict_to_common_indices",
    "select_day",
    "iterate_days",
    "adjust_series_forecast_timestamps",
]


def series_to_dataframe(y: pd.Series, name: str | None = None) -> pd.DataFrame:
    """Convert a Series to a DataFrame. The Series must have a name that is a string,
    or ``name`` must be provided,
    otherwise a ``ValueError`` is raised."""

    if name is None and isinstance(y.name, str):
        name = y.name

    if name is None:
        raise ValueError(
            f"pd.Series y should have a proper name (a string). "
            f"Found {y.name=} (type={type(y.name)})."
        )

    return y.to_frame(name)


def validate_column_names_as_string(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the labels of the columns are strings. If some labels are not
    strings, a ``ValueError`` is raised."""
    df.columns = [str(c) if isinstance(c, np.str_) else c for c in df.columns]
    wrong_labels = [(c, type(c)) for c in df.columns if not isinstance(c, str)]
    if wrong_labels:
        raise ValueError(f"Columns labels must be strings. Found {wrong_labels=}.")
    return df


@validate_call(config={"arbitrary_types_allowed": True})
def numpy_array_to_dataframe(
    array: np.ndarray, columns: list[str], index: pd.Index
) -> pd.DataFrame:
    """Transform a numpy array to a DataFrame. The array must have 1 or 2 dimensions,
    otherwise a ``ValueError`` is raised.

    Args:
        array: Input numpy array.
        columns: List of column names, must have length of ``array.shape[1]``.
        index: Index of the DataFrame, must have length of ``array.shape[0]``.

    Returns:
        DataFrame with the array values, with columns and index as specified.
    """
    if array.ndim == 1:
        array = np.reshape(array, (-1, 1))
    elif array.ndim != 2:
        raise ValueError(f"Array should have 1 or 2 dimensions. Found {array.ndim=}.")

    return pd.DataFrame(array, columns=columns, index=index)


def maybe_reorder_columns(
    X: pd.DataFrame, reference: Union[pd.DataFrame, Iterable], copy: bool = True
) -> pd.DataFrame:
    """Maybe reorder the columns of DataFrame ``X``. Reordering happens only if the
    columns of X are a subset of what is found in ``reference``.

    Args:
        X: Input DataFrame.
        reference: Reference DataFrame or iterable of column names.
        copy: If True, return a copy of the DataFrame. If False, return the DataFrame
            with the columns reordered inplace. Defaults to True.

    Returns:
        DataFrame with maybe reordered columns.
    """
    if isinstance(reference, pd.DataFrame):
        reference = reference.columns.to_list()

    columns = maybe_reorder_like(X.columns, reference)
    X = X[columns]
    if copy:
        return X.copy()
    return X


@validate_call(config={"arbitrary_types_allowed": True})
def find_datetime_index_unique_frequency(
    data: Union[pd.DataFrame, pd.Series],
) -> Union[pd.Timedelta, None]:
    """Find the unique frequency of a time series DataFrame or Series if it exists. A
    time series DataFrame or Series is expected to have a pd.DatetimeIndex. The frequency
    is the difference between two consecutive timestamps. If the frequency is not unique,
    None is returned.

    If the DataFrame or Series has not a pd.DatetimeIndex, None is returned.

    Args:
        data: The time series DataFrame or Series.

    Returns:
        The unique frequency if it exists, otherwise None.
    """
    try:
        indices = (data.index[1:] - data.index[:-1]).unique()
        if len(indices) == 1 and isinstance(indices[0], pd.Timedelta):
            return indices[0]
    except Exception:
        pass
    return None


def find_common_indices(*dfs: Iterable[pd.DataFrame]) -> pd.Index:
    """Find the common indices of multiple dataframes."""
    indices = None
    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"All inputs must be pandas DataFrames. Found {type(df)}.")
        if indices is None:
            indices = df.index
        else:
            indices = indices.intersection(df.index)
    return indices


def restrict_to_common_indices(*dfs: Iterable[pd.DataFrame]) -> Tuple[pd.DataFrame]:
    """Return the dataframes with only the common indices."""
    common_index = find_common_indices(*dfs)
    return tuple(df.loc[common_index] for df in dfs)


@validate_call(config={"arbitrary_types_allowed": True})
def select_day(df: pd.DataFrame, day: Union[pd.Timestamp, str]) -> pd.DataFrame:
    """Return a slice of the dataframe for a specific day."""
    day = pd.Timestamp(day)
    next_day = day + pd.Timedelta(days=1)

    day = day.strftime("%Y-%m-%d")
    next_day = next_day.strftime("%Y-%m-%d")

    return df.loc[(df.index >= day) & (df.index < next_day)]


def iterate_days(
    *dfs: Union[Iterable[pd.DataFrame], pd.DataFrame],
) -> Iterable[Union[Tuple[pd.DataFrame], pd.DataFrame]]:
    """Iterate over the dataframes day by day. If a tuple of dataframes is provided,
    the function will iterate over all dataframes and return a tuple of dataframes for
    each iteration. If a single dataframe is provided, the function will return a single
    dataframe for each iteration."""
    if len(dfs) == 1:
        dfs = dfs[0]

    if isinstance(dfs, pd.DataFrame):
        indices = dfs.index
    else:
        indices = find_common_indices(*dfs)

    for day in indices.strftime("%Y-%m-%d").unique():
        if isinstance(dfs, pd.DataFrame):
            yield select_day(dfs, day)
        else:
            yield tuple(select_day(df, day) for df in dfs)


@validate_call(config={"arbitrary_types_allowed": True})
def adjust_series_forecast_timestamps(data: pd.Series) -> pd.Series:
    """Transform a forecast Series results to a Series with timestamps.

    .. code-block::

        # From this
        Power [kW]_future_900      0.0
        Power [kW]_future_1800     0.0
        ...
        Power [kW]_future_14400    0.0
        Name: 2024-04-01 00:00:00, dtype: float64

        # To this
        2024-04-01 00:15:00    0.0
        2024-04-01 00:30:00    0.0
        ...
        2024-04-01 04:00:00    0.0
        Name: Power [kW], dtype: float64


    Args:
        data: The forecast Series.

    Returns:
        The transformed Series.
    """
    if not all(["_future_" in i for i in data.index]):
        raise ValueError("The data must have '_future_' in the index!")

    names = np.unique([i.split("_future")[0] for i in data.index])
    if len(names) > 1:
        raise ValueError(
            f"The data must have only one unique root name! Found: {names}"
        )

    date = data.name
    if not isinstance(date, pd.Timestamp):
        raise ValueError("The data name must be a pandas Timestamp!")
    timedeltas = [pd.Timedelta(seconds=int(c.split("_")[-1])) for c in data.index]
    timedeltas = pd.to_timedelta(timedeltas)

    results = pd.Series(data=data.values, index=date + timedeltas, name=names[0])
    return results
