from copy import deepcopy
from typing import Annotated, List

import pandas as pd
from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator

__all__ = [
    "TimeDataFrame",
    "CompleteTimeDataFrame",
    "WeekTimeDataFrame",
    "DayTimeDataFrame",
    "HourTimeDataFrame",
    "MinuteTimeDataFrame",
    "SecondTimeDataFrame",
]


class TimeDataFrameValidator(BaseModel):
    """Validator for a time series DataFrame. The type checking is implemented in the
    ``__call__`` method. If the DataFrame index is a proper DateTimeIndex, returns a
    DataFrame where:

    * The index is sorted;
    * The index name is set to ``'datetime'``.

    If the DataFrame index is not a proper DateTimeIndex, candidates columns as
    specified in ``allowed_columns`` are checked case-insensitively one by one. If the
    DataFrame contains a columns that is of type datetime (checked with
    ``pd.api.types.is_datetime64_any_dtype``), returns a DataFrame where:

    * The matched column is set as index and the old index is dropped;
    * The new index is sorted;
    * The new index name is set to ``'datetime'``.

    Args:
        allowed_columns: Columns that are allowed to be checked to contain a time index.
         If ``None``, ``allowed_columns`` is set to ``["timestamp", "datetime", "date"]``.
         Defaults to None.
    """

    allowed_columns: List[str] | None = None

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"df should be a pd.DataFrame, found {type(df)=}.")

        if pd.api.types.is_datetime64_any_dtype(df.index):
            # Needed?
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.DatetimeIndex(df.index)
            df = df.sort_index()
            df.index.names = ["datetime"]
            return df.sort_index()

        allowed_columns = deepcopy(self.allowed_columns)
        if allowed_columns is None:
            allowed_columns = ["timestamp", "datetime", "date"]

        case_insensitive_mapping = {str(c).lower(): c for c in df.columns.values}
        for name in allowed_columns:
            column = case_insensitive_mapping.get(name.lower(), None)
            if column is not None:
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df = df.set_index(column, drop=True).sort_index()
                    df.index.names = ["datetime"]
                    return df

        raise ValueError(
            f"Couldn't find a proper datetime column, tried {allowed_columns})."
        )


class CompleteTimeDataFrameValidator(TimeDataFrameValidator):
    """Validator for compluete time series DataFrame. It extends the functionality
    of ``TimeDataFrameValidator`` (subclass) by ensuring that the time index has a unique
    frequency. If ``allowed_frequency`` is provided, the unique frequency is checked
    to correspond to ``allowed_frequency``.

    Args:
        allowed_columns: Columns that are allowed to be checked to contain a time index.
         If ``None``, ``allowed_columns`` is set to ``["timestamp", "datetime", "date"]``.
         Defaults to None.

        allowed_frequency: If not None, the unique frequency is checked to match
         ``allowed_frequency``. If the match fail, a ``ValueError`` is raise, which
         is caught by Pydantic. Defaults to None.
    """

    allowed_frequency: pd.Timedelta | None = None

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().__call__(df)

        timedeltas = (df.index[1:] - df.index[:-1]).unique()
        if len(timedeltas) != 1:
            raise ValueError(f"Found different frequencies in DataFrame: {timedeltas}")

        if self.allowed_frequency is not None:
            if timedeltas[0] != self.allowed_frequency:
                raise ValueError(
                    f"Found frequency {timedeltas[0]} but {self.allowed_frequency=}."
                )

        return df


# PositiveInt = Annotated[int, AfterValidator(validate_positive_int)]


#: Custom annotated type for time series DataFrame.
TimeDataFrame = Annotated[pd.DataFrame, AfterValidator(TimeDataFrameValidator())]

#: Custom annotated type for contiguous time series DataFrame.
CompleteTimeDataFrame = Annotated[
    pd.DataFrame, AfterValidator(CompleteTimeDataFrameValidator())
]

#: Custom annotated type for contiguous time series DataFrame whose frequency is weekly,
#: i.e., the time delta between successive rows is 7 days.
WeekTimeDataFrame = Annotated[
    pd.DataFrame,
    AfterValidator(
        CompleteTimeDataFrameValidator(allowed_frequency=pd.Timedelta(days=7))
    ),
]

#: Custom annotated type for contiguous time series DataFrame whose frequency is daily,
#: i.e., the time delta between successive rows is 1 day.
DayTimeDataFrame = Annotated[
    pd.DataFrame,
    AfterValidator(
        CompleteTimeDataFrameValidator(allowed_frequency=pd.Timedelta(days=1))
    ),
]

#: Custom annotated type for contiguous time series DataFrame whose frequency is hourly,
#: i.e., the time delta between successive rows is 1 hour.
HourTimeDataFrame = Annotated[
    pd.DataFrame,
    AfterValidator(
        CompleteTimeDataFrameValidator(allowed_frequency=pd.Timedelta(hours=1))
    ),
]

#: Custom annotated type for contiguous time series DataFrame whose frequency is minute,
#: i.e., the time delta between successive rows is 1 minute.
MinuteTimeDataFrame = Annotated[
    pd.DataFrame,
    AfterValidator(
        CompleteTimeDataFrameValidator(allowed_frequency=pd.Timedelta(minutes=1))
    ),
]

#: Custom annotated type for contiguous time series DataFrame whose frequency is second,
#: i.e., the time delta between successive rows is 1 second.
SecondTimeDataFrame = Annotated[
    pd.DataFrame,
    AfterValidator(
        CompleteTimeDataFrameValidator(allowed_frequency=pd.Timedelta(seconds=1))
    ),
]
