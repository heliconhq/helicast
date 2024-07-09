from typing import Annotated, Literal, Union

import pandas as pd
from pydantic import BeforeValidator
from typing_extensions import Self

from helicast.base import HelicastBaseEstimator, InvertibleTransformerMixin, dataclass
from helicast.column_filters._base import ColumnFilter
from helicast.sklearn import helicast_auto_wrap
from helicast.utils._dataframe import maybe_reorder_columns

__all__ = ["FilteredTransformer"]


@dataclass
class FilteredTransformer(InvertibleTransformerMixin, HelicastBaseEstimator):
    """Meta-transformer that applies a transformer only to a subset of columns.

    Args:
        transformer: Transformer to apply to the filtered columns.
        filter: ColumnFilter instance to select the columns to which the transformer
            will be applied.
        remainder: What to do with the columns that are not transformed. If
            "passthrough", they will be concatenated to the transformed columns. If
            "drop", they will be dropped. Defaults to "passthrough".
    """

    transformer: Annotated[HelicastBaseEstimator, BeforeValidator(helicast_auto_wrap)]
    filter: ColumnFilter

    remainder: Literal["passthrough", "drop"] = "passthrough"

    def _fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> Self:
        self.transformer.fit(self.filter.fit_transform(X), y, **kwargs)
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        columns = self.filter(X)
        X_tr = self.transformer.transform(X[columns])
        if self.remainder == "passthrough":
            for c in X.columns:
                if c not in columns:
                    X_tr[c] = X[c]
        return maybe_reorder_columns(X_tr, X)

    def _inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self.transformer, "inverse_transform"):
            raise ValueError(
                f"Transformer {self.transformer.__class__.__name__} does not "
                f"have an inverse_transform method."
            )
        columns = self.transformer.feature_names_out_
        X_inv_tr = self.transformer.inverse_transform(X[columns])
        if self.remainder == "passthrough":
            for c in X.columns:
                if c not in columns:
                    X_inv_tr[c] = X[c]
        return maybe_reorder_columns(X_inv_tr, X)
