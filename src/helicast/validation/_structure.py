from typing import ClassVar

import pandas as pd
from typing_extensions import Self

from helicast.base import (
    HelicastBaseEstimator,
    InvertibleTransformerMixin,
    dataclass,
)

__all__ = [
    "ColumnNamesValidator",
]


@dataclass
class ColumnNamesValidator(InvertibleTransformerMixin, HelicastBaseEstimator):
    """Invertible transformer that checks if the columns names of a DataFrame match the
    ones seen during fit. The DataFrame returned by ``transform`` and
    ``inverse_transform`` will have the same ordering as the one seen during fit."""

    # Private flag to disable the X column subset filtering during validation
    _disable_X_column_subset_filtering: ClassVar[bool] = True

    def _fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None, **kwargs
    ) -> Self:
        self.columns_ = X.columns.tolist()
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.columns_) - set(X.columns)
        extra = set(X.columns) - set(self.columns_)
        if missing or extra:
            raise ValueError(
                f"Columns do not match. Missing: {missing}, Extra: {extra}."
            )

        return X[self.columns_]

    def _inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform(X)
