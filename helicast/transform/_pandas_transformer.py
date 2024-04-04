import logging
from typing import List, Literal

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from typing_extensions import Self

from helicast.column_filters._base import AllSelector, ColumnFilter
from helicast.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

__all__ = ["PandasTransformer"]


class PandasTransformer(BaseEstimator):
    def __init__(
        self,
        transformer: BaseEstimator,
        filter: ColumnFilter,
        remainder: Literal["passthrough", "drop"] = "passthrough",
        extra: Literal["ignore", "warn", "raise"] = "warn",
        extra_ignored: Literal["ignore", "warn", "raise"] = "warn",
        missing_ignored: Literal["ignore", "warn", "raise"] = "warn",
    ) -> None:

        self.transformer = transformer
        self.filter = filter
        self.remainder = remainder
        self.extra = extra
        self.extra_ignored = extra_ignored
        self.missing_ignored = missing_ignored

    def _get_extra_columns_report(self, selected_columns: List[str]) -> str:
        """Get a string message if there are columns in ``selected_columns`` that
        would have been fitted during fit but were not there. If no such columns exists,
        returns an empty string."""
        columns = set(selected_columns) - set(self.feature_names_in_)
        msg = ""
        if columns:
            msg = (
                f"Columns {columns} have not been seen during fit but would "
                "have been fitted."
            )
        return msg

    def _get_missing_columns_report(self, selected_columns: List[str]) -> str:
        """Get a string message if `selected_columns`` do not contain all the columns
        that have been seen and fitted during fit. If no such columns exists, returns an
        empty string."""
        columns = set(self.feature_names_in_) - set(selected_columns)
        msg = ""
        if columns:
            msg = f"Columns {columns} have been seen during fit but are now missing!"
        return msg

    def _get_extra_ignored_columns_report(self, ignored_columns: List[str]) -> str:
        """Get a string message if there are columns in ``ignored_columns`` that would
        have been ignored during fit but were not there. If no such columns exists,
        returns an empty string."""
        columns = set(ignored_columns) - set(self.ignored_feature_names_in_)
        msg = ""
        if columns:
            msg = (
                f"Columns {columns} are ignored but have not been seen during fit (in "
                "which case, they would have been ignored anyway)."
            )
        return msg

    def _get_missing_ignored_columns_report(self, ignored_columns: List[str]) -> str:
        """Get a string message if `ignored_columns`` does not contain all the
        columns that have been seen and ignored during fit. If no such columns exists,
        returns an empty string."""
        columns = set(self.ignored_feature_names_in_) - set(ignored_columns)
        msg = ""
        if columns:
            msg = (
                f"Columns {columns} have been ignored during fit, but are now missing."
            )
        return msg

    def _validate_columns(
        self, selected_columns: List[str], ignored_columns: List[str]
    ):
        reports = [
            (self._get_missing_columns_report(selected_columns), "raise"),
            (self._get_extra_columns_report(selected_columns), self.extra),
            (
                self._get_missing_ignored_columns_report(ignored_columns),
                self.missing_ignored,
            ),
            (
                self._get_extra_ignored_columns_report(ignored_columns),
                self.extra_ignored,
            ),
        ]

        for i, j in reports:
            if i:
                match j:
                    case "ignore":
                        pass
                    case "warn":
                        logger.warning(i)
                    case "raise":
                        raise KeyError(i)
                    case _:
                        raise ValueError(
                            f"Not valid {j=}. Should be 'ignore', 'warn' or 'raise'."
                        )

    def fit(self, X: pd.DataFrame, y=None) -> Self:

        self.all_feature_names_in_ = X.columns.to_list()
        self.feature_names_in_ = self.filter(X)
        self.ignored_feature_names_in_ = [
            i for i in self.all_feature_names_in_ if i not in self.feature_names_in_
        ]

        name = f"{self.transformer.__class__.__name__}"

        self.column_transformer_ = ColumnTransformer(
            [(name, self.transformer, AllSelector())],
            remainder=self.remainder,
            verbose=False,
            verbose_feature_names_out=False,
        )

        self.column_transformer_.fit(X[self.feature_names_in_], y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        index = X.index
        columns = X.columns.to_list()
        selected_columns = self.filter(X)
        ignored_columns = list(set(columns) - set(selected_columns))

        self._validate_columns(selected_columns, ignored_columns)

        new_X = self.column_transformer_.transform(X[self.feature_names_in_])
        new_X = pd.DataFrame(
            new_X, columns=self.column_transformer_.get_feature_names_out(), index=index
        )

        if self.remainder == "passthrough":
            for c in ignored_columns:
                new_X.loc[:, c] = X[c]

        return new_X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
