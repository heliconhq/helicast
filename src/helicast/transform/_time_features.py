import logging
from typing import Literal, Optional, Union

import holidays
import pandas as pd

from helicast.base import BaseEstimator, StatelessTransformerMixin, dataclass
from helicast.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

__all__ = [
    "HourColumnAdder",
    "add_hour_column",
    "DayOfWeekColumnAdder",
    "add_day_of_week_column",
    "DayOfYearColumnAdder",
    "add_day_of_year_column",
    "SwedenPublicHolidayColumnAdder",
    "add_sweden_public_holiday_column",
]


def _validate_time_index(
    df: pd.DataFrame, timestamp_column: Optional[str] = None
) -> pd.DatetimeIndex:
    if timestamp_column is None:
        time_index = df.index
    else:
        time_index = df[timestamp_column]

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
class HourColumnAdder(StatelessTransformerMixin, BaseEstimator):
    timestamp_column: Optional[str] = None

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        time_index = _validate_time_index(X, timestamp_column=self.timestamp_column)

        X_new = X.copy()
        X_new["hour"] = time_index.hour.astype(int)

        return X_new


def add_hour_column(
    df: pd.DataFrame, timestamp_column: Optional[str] = None
) -> pd.DataFrame:
    return HourColumnAdder(timestamp_column=timestamp_column).fit_transform(df)


@dataclass
class DayOfWeekColumnAdder(StatelessTransformerMixin, BaseEstimator):
    timestamp_column: Optional[str] = None

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        time_index = _validate_time_index(X, timestamp_column=self.timestamp_column)

        X_new = X.copy()
        X_new["day_of_week"] = time_index.day_of_week.astype(int)

        return X_new


def add_day_of_week_column(
    df: pd.DataFrame, timestamp_column: Optional[str] = None
) -> pd.DataFrame:
    return DayOfWeekColumnAdder(timestamp_column=timestamp_column).fit_transform(df)


@dataclass
class DayOfYearColumnAdder(StatelessTransformerMixin, BaseEstimator):
    timestamp_column: Optional[str] = None

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        time_index = _validate_time_index(X, timestamp_column=self.timestamp_column)

        X_new = X.copy()
        X_new["day_of_year"] = time_index.day_of_year.astype(int)

        return X_new


def add_day_of_year_column(
    df: pd.DataFrame, timestamp_column: Optional[str] = None
) -> pd.DataFrame:
    return DayOfYearColumnAdder(timestamp_column=timestamp_column).fit_transform(df)


@dataclass
class SwedenPublicHolidayColumnAdder(StatelessTransformerMixin, BaseEstimator):
    timestamp_column: Optional[str] = None

    encoding: Literal["int", "bool", "str"] = "int"

    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        time_index = _validate_time_index(X, timestamp_column=self.timestamp_column)

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


def add_sweden_public_holiday_column(
    df: pd.DataFrame,
    timestamp_column: Optional[str] = None,
    encoding: Literal["int", "bool", "str"] = "int",
) -> pd.DataFrame:
    tr = SwedenPublicHolidayColumnAdder(
        timestamp_column=timestamp_column, encoding=encoding
    )
    return tr.fit_transform(df)
