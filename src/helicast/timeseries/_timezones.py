"""Sub-module containing classes and functions that can be used to handle the timezone of
a time series pd.DataFrame."""

from typing import Any

import pandas as pd

from helicast.base import HelicastBaseEstimator, StatelessTransformerMixin, dataclass
from helicast.utils._docs import link_docs_to_class

__all__ = [
    "TzDatetimeIndexLocalizator",
    "tz_localize_datetime_index",
    "TzDatetimeIndexConverter",
    "tz_convert_datetime_index",
    "TzDatetimeIndexTransformer",
    "tz_transform_datetime_index",
]


@dataclass
class TzDatetimeIndexLocalizator(StatelessTransformerMixin, HelicastBaseEstimator):
    """TZ-localize the index of a DataFrame to a specific timezone. The index must be
    a DatetimeIndex that is timezone naive.

    Args:
        tz: Timezone to localize the index to.
    """

    tz: Any

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(f"Index must be a DatetimeIndex. Found: {type(X.index)}.")

        if X.index.tz is not None:
            raise ValueError(
                f"Index must be timezone naive if you want to localize your index. "
                f"Found timezone aware index with timezone: {X.index.tz}. Maybe you "
                f"wanted to convert the timezone: use TzDatetimeIndexConverter."
            )

        X = X.copy()
        X.index = X.index.tz_localize(self.tz)
        return X


@link_docs_to_class(cls=TzDatetimeIndexLocalizator)
def tz_localize_datetime_index(X: pd.DataFrame, tz: Any) -> pd.DataFrame:
    return TzDatetimeIndexLocalizator(tz=tz).fit_transform(X)


@dataclass
class TzDatetimeIndexConverter(StatelessTransformerMixin, HelicastBaseEstimator):
    """TZ-convert the index of a DataFrame to a specific timezone. The index must be
    a DatetimeIndex that is timezone aware.

    Args:
        tz: Timezone to convert the index to.
    """

    tz: Any

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(f"Index must be a DatetimeIndex. Found: {type(X.index)}.")

        if X.index.tz is None:
            raise ValueError(
                "Index must be timezone aware if you want to tz_convert your index, "
                "found timezone naive index. Maybe you wanted to tz_localize: use "
                "TzDatetimeIndexLocalizator. "
            )

        X = X.copy()
        X.index = X.index.tz_convert(self.tz)
        return X


@link_docs_to_class(cls=TzDatetimeIndexConverter)
def tz_convert_datetime_index(X: pd.DataFrame, tz: Any) -> pd.DataFrame:
    return TzDatetimeIndexConverter(tz=tz).fit_transform(X)


@dataclass
class TzDatetimeIndexTransformer(StatelessTransformerMixin, HelicastBaseEstimator):
    """Localize or convert the index of a DataFrame to a specific timezone. The index
    must be a DatetimeIndex (both timezone aware and naive are accepted).

    Args:
        tz: Timezone to localize the index to.
    """

    tz: Any

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(f"Index must be a DatetimeIndex. Found: {type(X.index)}.")

        if X.index.tz is None:
            return TzDatetimeIndexLocalizator(tz=self.tz).transform(X)
        else:
            return TzDatetimeIndexConverter(tz=self.tz).transform(X)


@link_docs_to_class(cls=TzDatetimeIndexTransformer)
def tz_transform_datetime_index(X: pd.DataFrame, tz: Any) -> pd.DataFrame:
    return TzDatetimeIndexTransformer(tz=tz).fit_transform(X)
