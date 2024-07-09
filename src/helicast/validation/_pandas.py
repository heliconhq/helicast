from typing import Optional, Tuple, Union

import pandas as pd
from pydantic import validate_call

from helicast.transform import rename_index, sort_index

_VALIDATE_CONFIG = {"arbitrary_types_allowed": True, "validate_return": True}

__all__ = [
    "auto_set_datetime_index",
    "validate_no_duplicated_index",
    "validate_datetime_index",
    "validate_tz_aware_datetime_index",
    "tz_localize_index",
    "tz_convert_index",
    "tz_convert_or_localize_index",
]


@validate_call(config=_VALIDATE_CONFIG)
def auto_set_datetime_index(
    obj: pd.DataFrame,
    first_try_columns: Optional[Tuple] = ("datetime", "date"),
) -> pd.DataFrame:
    """Set the index of ``obj`` to the first column that is a datetime column. The
    columns provided in ``first_try_columns`` will be tried first. If the index is
    already a DatetimeIndex, it will be returned as is.

    Args:
        obj: Pandas DataFrame to set the index of.
        first_try_columns: Tuple of column names to try first. If None, no columns will
            be tried first. Defaults to ("datetime", "date").

    Raises:
        ValueError: If the index is not a DatetimeIndex and no datetime columns are found.

    Returns:
        Pandas DataFrame with a DatetimeIndex.
    """
    if isinstance(obj.index, pd.DatetimeIndex):
        return obj

    if first_try_columns is None:
        first_try_columns = list()
    else:
        first_try_columns = list(first_try_columns)

    datetime_columns = obj.select_dtypes(include=["datetime", "datetimetz"]).columns
    if len(datetime_columns) == 0:
        raise ValueError("No datetime columns found.")

    first_try_columns = [c for c in first_try_columns if c in datetime_columns]

    if len(first_try_columns) > 0:
        return obj.set_index(first_try_columns[0])
    return obj.set_index(datetime_columns[0])


@validate_call(config=_VALIDATE_CONFIG)
def validate_no_duplicated_index(
    obj: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    """Validate that the index of ``obj`` does not contain duplicated values.

    Args:
        obj: Pandas DataFrame or Series to validate.

    Raises:
        ValueError: If the index contains duplicated values.

    Returns:
        The input object with the index validated.
    """
    duplicates = obj.index[obj.index.duplicated()].unique()
    if len(duplicates) > 0:
        raise ValueError(f"Index contains duplicated values. Found: {duplicates}.")
    return obj


@validate_call(config=_VALIDATE_CONFIG)
def validate_datetime_index(
    obj: Union[pd.DataFrame, pd.Series],
    frequency: Union[bool, pd.Timedelta] = False,
    tz_aware: bool = False,
    sort: bool = True,
    name: Optional[str] = "datetime",
    allow_duplicated_index: bool = False,
) -> Union[pd.DataFrame, pd.Series]:
    """Validate that the index of ``obj`` is a DatetimeIndex with the specified
    properties.

    Args:
        obj: Pandas DataFrame or Series to validate.
        frequency: Frequency of the index. If False, the frequency will not be checked
            and irregular indexes will be allowed. If True, the frequency will be
            checked and only regular indexes will be allowed. If a pd.Timedelta, the
            frequency will be checked and the index must have this frequency. Defaults
            to False.
        tz_aware: If True, the index must be timezone aware. Defaults to False.
        sort: If True, the index will be sorted. Note that if frequency is not False,
            the index will be sorted regardless of this parameter. Defaults to True.
        name: Name to give to the index. If None, no name is given. Defaults to "datetime".
        allow_duplicated_index: If True, duplicated index values are allowed. Defaults
            to False.

    Raises:
        ValueError: If the index is not a DatetimeIndex.
        ValueError: If the index contains duplicated values and allow_duplicated_index is
            False.
        ValueError: If the index is not timezone aware and tz_aware is True.
        ValueError: If the index has multiple frequencies and frequency is True.
        ValueError: If frequency is a pd.Timedelta and the index does not have this
            frequency.

    Returns:
        The input object with the index validated.
    """
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise ValueError(f"Index must be a DatetimeIndex. Found: {type(obj.index)}.")

    if not allow_duplicated_index:
        obj = validate_no_duplicated_index(obj)

    if sort or frequency is not False:
        obj = sort_index(obj)

    if frequency is True or isinstance(frequency, pd.Timedelta):
        timedeltas = obj.index.to_series().diff()[1:].unique()
        if len(timedeltas) > 1:
            raise ValueError(
                f"Index must have a unique frequency. Found: {timedeltas}."
            )
        if isinstance(frequency, pd.Timedelta):
            if timedeltas[0] != frequency:
                raise ValueError(
                    f"Expected frequency: {frequency}. Found: {timedeltas[0]}"
                )

    if tz_aware and obj.index.tz is None:
        raise ValueError("Index must be timezone aware, found timezone naive index.")

    if name is not None:
        obj = rename_index(obj, name)

    return obj


@validate_call(config=_VALIDATE_CONFIG)
def validate_tz_aware_datetime_index(
    obj: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise ValueError(f"Index must be a DatetimeIndex. Found: {type(obj.index)}.")

    if obj.index.tz is None:
        raise ValueError("Index must be timezone aware, found timezone naive index.")
    return obj


@validate_call(config=_VALIDATE_CONFIG)
def tz_localize_index(
    obj: Union[pd.DataFrame, pd.Series], tz: str
) -> Union[pd.DataFrame, pd.Series]:
    """Timezone-localize the index of ``obj`` to timezone ``tz``.

    Args:
        obj: Pandas DataFrame or Series to localize the index of.
        tz: Timezone to localize the index to.

    Raises:
        ValueError: If the index is not a DatetimeIndex.
        ValueError: If the index is timezone aware.

    Returns:
        Pandas DataFrame or Series with the index timezone localized.
    """
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise ValueError(f"Index must be a DatetimeIndex. Found: {type(obj.index)}.")
    if obj.index.tz is not None:
        raise ValueError(
            f"Index must be timezone naive if you want to tz_localize your index. "
            f"Found timezone aware index with timezone: {obj.index.tz}. "
            f"Maybe you wanted to tz_convert: use TZConvertIndexValidator. "
        )
    obj.index = obj.index.tz_localize(tz)
    return obj


@validate_call(config=_VALIDATE_CONFIG)
def tz_convert_index(
    obj: Union[pd.DataFrame, pd.Series], tz: str
) -> Union[pd.DataFrame, pd.Series]:
    """Timezone-convert the index of ``obj`` to timezone ``tz``.

    Args:
        obj: Pandas DataFrame or Series to convert the index of.
        tz: Timezone to convert the index to.

    Raises:
        ValueError: If the index is not a DatetimeIndex.
        ValueError: If the index is timezone naive.

    Returns:
        Pandas DataFrame or Series with the index timezone converted.
    """
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise ValueError(f"Index must be a DatetimeIndex. Found: {type(obj.index)}.")
    if obj.index.tz is None:
        raise ValueError(
            "Index must be timezone aware if you want to tz_convert your index. "
            "Maybe you wanted to tz_localize: use TZLocalizeIndexValidator. "
        )
    obj.index = obj.index.tz_convert(tz)
    return obj


@validate_call(config=_VALIDATE_CONFIG)
def tz_convert_or_localize_index(
    obj: Union[pd.DataFrame, pd.Series], tz: str
) -> Union[pd.DataFrame, pd.Series]:
    """Timezone-convert or localize the index of ``obj`` to timezone ``tz``.

    Args:
        obj: Pandas DataFrame or Series to convert or localize the index of.
        tz: Timezone to convert or localize the index to.

    Raises:
        ValueError: If the index is not a DatetimeIndex.

    Returns:
        Pandas DataFrame or Series with the index timezone converted or localized.
    """
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise ValueError(f"Index must be a DatetimeIndex. Found: {type(obj.index)}.")
    if obj.index.tz is None:
        obj.index = obj.index.tz_localize(tz)
    else:
        obj.index = obj.index.tz_convert(tz)
    return obj
