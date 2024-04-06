import logging
from typing import Annotated, List, Literal, Union

import pandas as pd
from pydantic import AfterValidator, Field, model_validator
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from typing_extensions import Self

from helicast.base import PydanticBaseEstimator, _validate_X_y
from helicast.column_filters._base import AllSelector, ColumnFilter
from helicast.logging import configure_logging
from helicast.typing import UNSET

configure_logging()
logger = logging.getLogger(__name__)


__all__ = ["PandasTransformer"]


class PandasTransformer(PydanticBaseEstimator):

    transformer: BaseEstimator
    filter: Union[ColumnFilter, None] = None

    remainder: Literal["passthrough", "drop"] = "passthrough"
    extra: Literal["ignore", "warn", "raise"] = "raise"
    extra_ignored: Literal["ignore", "warn", "raise"] = "raise"
    missing_ignored: Literal["ignore", "warn", "raise"] = "raise"

    #: All the features seen during fit
    all_feature_names_in_: List[str] = Field(UNSET, init=False, repr=False)

    #: All the relevant features seen during fit. It's a subset of `all_feature_names_in_`
    #: after the `self.filter` has been used
    feature_names_in_: List[str] = Field(UNSET, init=False, repr=False)

    #: All the features seen during fit but ignored. This is disjoint with `feature_names_in`
    #: and the union of `ignored_feature_names_in_` and `feature_names_in` will give
    #: `all_feature_names_in_`
    ignored_feature_names_in_: List[str] = Field(UNSET, init=False, repr=False)

    # TODO: This should not be needed?
    target_names_in_: Union[List[str], None] = Field(UNSET, init=False, repr=False)

    @model_validator(mode="after")
    def _check_transformer(self) -> "PandasTransformer":
        if not hasattr(self.transformer, "transform"):
            raise TypeError(f"{self.transformer=} has not `transform` method!")
        return self

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

        X, y = _validate_X_y(X, y)

        self.all_feature_names_in_ = X.columns.to_list()
        if self.filter is None:
            self.feature_names_in_ = self.all_feature_names_in_[:]
        else:
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
        if self.filter is None:
            selected_columns = columns[:]
        else:
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

        # Keep the original ordering only if the columns are identical
        if set(columns) == set(new_X.columns):
            new_X = new_X[columns]

        return new_X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
