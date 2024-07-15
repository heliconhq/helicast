"""Sub-module containing classes and functions to validate pandas time series data. The
time series nature of the pd.DataFrame is validated by checking that the index is of
time DatetimeIndex."""

from typing import Literal, Union

import pandas as pd

from helicast.base import HelicastBaseEstimator, StatelessTransformerMixin, dataclass
from helicast.typing._time import Timedelta
from helicast.utils import are_timezones_equivalent
from helicast.utils._docs import link_docs_to_class

__all__ = [
    "DatetimeIndexValidator",
    "validate_datetime_index",
    "TzAwareDatetimeIndexValidator",
    "validate_tz_aware_datetime_index",
    "TzNaiveDatetimeIndexValidator",
    "validate_tz_naive_datetime_index",
]


@dataclass
class DatetimeIndexValidator(StatelessTransformerMixin, HelicastBaseEstimator):
    """Validate that a DataFrame has a DatetimeIndex as index

    Args:
        frequency: If True, check if the index has a unique frequency. If False, no
            frequency validation is performed. If a pd.Timedelta is provided, check if
            the index has the provided frequency. Defaults to False.
        tz: If "aware", check if the index is timezone aware. If "naive", check if the
            index is timezone naive. If a string is provided, check if the index is
            timezone aware with the provided timezone. If None, no timezone validation
            is performed. Defaults to None.
        index_name: If a string is provided, rename the index to the provided name.
            If None, do not rename the index. Defaults to None.
    """

    frequency: Union[bool, Timedelta] = False
    tz: Union[Literal["aware", "naive"], str, None] = None
    index_name: Union[str, None] = None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(f"Index must be a DatetimeIndex. Found: {type(X.index)}.")

        if self.frequency is True or isinstance(self.frequency, pd.Timedelta):
            timedeltas = X.index.to_series().diff()[1:].unique()
            if len(timedeltas) > 1:
                raise ValueError(
                    f"Index must have a unique frequency. Found: {timedeltas}."
                )
            if isinstance(self.frequency, pd.Timedelta):
                if timedeltas[0] != self.frequency:
                    raise ValueError(
                        f"Expected frequency: {self.frequency}. Found: {timedeltas[0]}"
                    )

        if self.tz == "aware" and X.index.tz is None:
            raise ValueError(
                "Index must be timezone aware, found timezone naive index."
            )
        elif self.tz == "naive" and X.index.tz is not None:
            raise ValueError(
                "Index must be timezone naive, found timezone aware index."
            )
        elif isinstance(self.tz, str):
            if X.index.tz is None:
                raise ValueError(
                    f"Index must be timezone aware with tz={self.tz}, "
                    f"found timezone naive index."
                )
            if not are_timezones_equivalent(self.tz, X.index.tz):
                raise ValueError(
                    f"Index timezone must be equivalent to {self.tz}, found {X.index.tz}."
                )

        if isinstance(self.index_name, str):
            X = X.rename_axis(self.index_name)

        return X.copy()


@link_docs_to_class(cls=DatetimeIndexValidator)
def validate_datetime_index(
    X: pd.DataFrame,
    frequency: Union[bool, Timedelta] = False,
    tz: Union[Literal["aware", "naive"], str, None] = None,
    index_name: Union[str, None] = None,
) -> pd.DataFrame:
    return DatetimeIndexValidator(
        frequency=frequency, tz=tz, index_name=index_name
    ).fit_transform(X)


@dataclass
class TzAwareDatetimeIndexValidator(StatelessTransformerMixin, HelicastBaseEstimator):
    """Validate that a DataFrame has a timezone-aware DatetimeIndex as index"""

    tz: Union[Literal["aware"], str] = "aware"

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return DatetimeIndexValidator(tz=self.tz).transform(X)


@link_docs_to_class(cls=TzAwareDatetimeIndexValidator)
def validate_tz_aware_datetime_index(
    X: pd.DataFrame, tz: Union[Literal["aware"], str]
) -> pd.DataFrame:
    return TzAwareDatetimeIndexValidator(tz=tz).fit_transform(X)


@dataclass
class TzNaiveDatetimeIndexValidator(StatelessTransformerMixin, HelicastBaseEstimator):
    """Validate that a DataFrame has a timezone-naive DatetimeIndex as index"""

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return DatetimeIndexValidator(tz="naive").transform(X)


@link_docs_to_class(cls=TzNaiveDatetimeIndexValidator)
def validate_tz_naive_datetime_index(X: pd.DataFrame) -> pd.DataFrame:
    return TzNaiveDatetimeIndexValidator().fit_transform(X)
