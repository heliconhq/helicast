from typing import Literal, Optional, Union

import holidays
import numpy as np
import pandas as pd

from helicast.base import (
    StatelessEstimator,
    TransformerMixin,
    dataclass,
)
from helicast.transform import FutureColumnsAdder
from helicast.utils import find_datetime_index_unique_frequency, link_docs_to_class

__all__ = [
    "ForecastHorizonColumnsAdder",
    "add_forecast_horizon_columns",
    "HourColumnAdder",
    "add_hour_column",
    "DayOfWeekColumnAdder",
    "add_day_of_week_column",
    "DayOfYearColumnAdder",
    "add_day_of_year_column",
    "SwedenPublicHolidayColumnAdder",
    "add_sweden_public_holiday_column",
]


@dataclass
class ForecastHorizonColumnsAdder(TransformerMixin, StatelessEstimator):
    """Add columns with future values of the input columns. The columns are
    labeled with the time difference in seconds from the input columns.

    Args:
        horizon: Time horizon to forecast.
        gap: Time gap between the input columns and the future columns. If None,
            the gap is 0. Defaults to None.
        dropna_boundary: If True, drop the rows with NaN values at the boundary
            of the DataFrame. Defaults to True.
    """

    horizon: pd.Timedelta
    gap: Union[pd.Timedelta, None] = None
    dropna_boundary: bool = True

    def _get_future_columns_adder(self, X: pd.DataFrame):
        self.freq_ = find_datetime_index_unique_frequency(X)
        if self.freq_ is None:
            raise ValueError(
                "The DataFrame must have pd.DatetimeIndex with a unique frequency!"
            )
        start = 1
        if self.gap is not None:
            start = 1 + int(self.gap / self.freq_)

        n_steps = int(self.horizon / self.freq_)
        shifts = np.arange(start, start + n_steps + 1)
        return FutureColumnsAdder(shifts=shifts, dropna_boundary=self.dropna_boundary)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        old_columns = X.columns.to_list()
        tr = self._get_future_columns_adder(X)
        X_tr = tr.fit_transform(X)

        def _rename(c: str) -> str:
            root, digit = "_".join(c.split("_")[:-1]), int(c.split("_")[-1])
            timedelta_in_seconds = int(digit * self.freq_.total_seconds())
            return f"{root}_{timedelta_in_seconds}"

        mapping = {c: _rename(c) for c in X_tr.columns if c not in old_columns}
        X_tr = X_tr.rename(columns=mapping)

        return X_tr


@link_docs_to_class(cls=ForecastHorizonColumnsAdder)
def add_forecast_horizon_columns(
    X: pd.DataFrame,
    horizon: pd.Timedelta,
    gap: Union[pd.Timedelta, None] = None,
    dropna_boundary: bool = True,
) -> pd.DataFrame:
    return ForecastHorizonColumnsAdder(
        horizon=horizon, gap=gap, dropna_boundary=dropna_boundary
    ).fit_transform(X)


def _validate_time_index(
    df: pd.DataFrame, datetime_column: Optional[str] = None
) -> pd.DatetimeIndex:
    if datetime_column is None:
        time_index = df.index
    else:
        time_index = df[datetime_column]

    if not pd.api.types.is_datetime64_any_dtype(time_index):
        raise TypeError(f"Wrong time index. Found {time_index.dtypes}")

    if isinstance(time_index, pd.Series):
        time_index = pd.DatetimeIndex(time_index)
    return time_index


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


@dataclass
class HourColumnAdder(TransformerMixin, StatelessEstimator):
    """Add a column ``"hour"`` to the DataFrame with the hour of the day. The index
    of the DataFrame is expected to be a pd.DatetimeIndex or a column with datetime
    values must be specified by the ``datetime_column`` argument.

    Args:
        datetime_column: Column with datetime values. If None, the index of the
            DataFrame is used. Defaults to None.
    """

    datetime_column: Optional[str] = None

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        time_index = _validate_time_index(X, datetime_column=self.datetime_column)

        X_new = X.copy()
        X_new["hour"] = time_index.hour.astype(int)

        return X_new


@link_docs_to_class(cls=HourColumnAdder)
def add_hour_column(
    X: pd.DataFrame, datetime_column: Optional[str] = None
) -> pd.DataFrame:
    return HourColumnAdder(datetime_column=datetime_column).fit_transform(X)


@dataclass
class DayOfWeekColumnAdder(TransformerMixin, StatelessEstimator):
    """Add a column ``"day_of_week"`` to the DataFrame with the day of the week as
    integer (0=Monday, 1=Tuesday, ..., 6=Sunday). The index of the DataFrame is expected
    to be a pd.DatetimeIndex or a column with datetime values must be specified by the
    ``datetime_column`` argument.

    Args:
        datetime_column: Column with datetime values. If None, the index of the
            DataFrame is used. Defaults to None.
    """

    datetime_column: Optional[str] = None

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        time_index = _validate_time_index(X, datetime_column=self.datetime_column)

        X_new = X.copy()
        X_new["day_of_week"] = time_index.day_of_week.astype(int)

        return X_new


@link_docs_to_class(cls=DayOfWeekColumnAdder)
def add_day_of_week_column(
    X: pd.DataFrame, datetime_column: Optional[str] = None
) -> pd.DataFrame:
    return DayOfWeekColumnAdder(datetime_column=datetime_column).fit_transform(X)


@dataclass
class DayOfYearColumnAdder(TransformerMixin, StatelessEstimator):
    """Add a column ``"day_of_year"`` to the DataFrame with the day of the year as
    integer. The index of the DataFrame is expected to be a pd.DatetimeIndex or a column
    with datetime values must be specified by the ``datetime_column`` argument.

    Args:
        datetime_column: Column with datetime values. If None, the index of the
            DataFrame is used. Defaults to None.
    """

    datetime_column: Optional[str] = None

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        time_index = _validate_time_index(X, datetime_column=self.datetime_column)

        X_new = X.copy()
        X_new["day_of_year"] = time_index.day_of_year.astype(int)

        return X_new


@link_docs_to_class(cls=DayOfYearColumnAdder)
def add_day_of_year_column(
    X: pd.DataFrame, datetime_column: Optional[str] = None
) -> pd.DataFrame:
    return DayOfYearColumnAdder(datetime_column=datetime_column).fit_transform(X)


@dataclass
class SwedenPublicHolidayColumnAdder(TransformerMixin, StatelessEstimator):
    """EXPERIMENTAL! Add a column ``"sweden_public_holiday"`` to the DataFrame with
    information about whether the day is a public holiday in Sweden. The index of the
    DataFrame is expected to be a pd.DatetimeIndex or a column with datetime values must
    be specified by the ``datetime_column`` argument.

    Args:
        datetime_column: Column with datetime values. If None, the index of the
            DataFrame is used. Defaults to None.
        encoding: Encoding of the column. Can be ``"int"`` (1 for public holiday, 0
            otherwise), ``"bool"`` (True for public holiday, False otherwise), or
            ``"str"`` (public holiday name). Defaults to ``"int"``.
    """

    datetime_column: Optional[str] = None

    encoding: Literal["int", "bool", "str"] = "int"

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        time_index = _validate_time_index(X, datetime_column=self.datetime_column)

        hd = _create_sweden_public_holiday_map(time_index.min(), time_index.max())

        X_new = X.copy()
        X_new["sweden_public_holiday"] = hd[time_index.dt.strftime("%Y-%m-%d")].values

        match self.encoding:
            case "int":
                X_new["sweden_public_holiday"] = (
                    X_new["sweden_public_holiday"].notna().astype(int)
                )
            case "bool":
                X_new["sweden_public_holiday"] = X_new["sweden_public_holiday"].notna()
            case "str":
                pass
            case _:
                raise ValueError()

        return X_new


@link_docs_to_class(cls=SwedenPublicHolidayColumnAdder)
def add_sweden_public_holiday_column(
    X: pd.DataFrame,
    datetime_column: Optional[str] = None,
    encoding: Literal["int", "bool", "str"] = "int",
) -> pd.DataFrame:
    tr = SwedenPublicHolidayColumnAdder(
        datetime_column=datetime_column, encoding=encoding
    )
    return tr.fit_transform(X)
