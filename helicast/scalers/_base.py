import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from enum import Flag, auto
from logging import getLogger
from typing import ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, validate_call
from sklearn.preprocessing import MaxAbsScaler
from typing_extensions import Self, Unpack

from helicast.logging import configure_logging
from helicast.typing import UNSET
from helicast.validation import HelicastBaseModel
from helicast.yaml import get_init_args, register_yaml

configure_logging()
logger = getLogger(__name__)

_VALIDATE_CONFIG = {
    "arbitrary_types_allowed": True,
    "validate_return": True,
    "strict": True,
}


class ColumnType(Flag):
    FLOAT = auto()
    INT = auto()
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    CATEGORY = auto()
    DATETIME = auto()
    ALL = auto()  # Special flag to select ALL the columns :)

    def __str__(self) -> str:
        name = super().__str__()
        return name.split(".")[-1].lower()


def filter_columns_by_type(df: pd.DataFrame, column_type: ColumnType) -> List[str]:
    """Get a list of column that correspond to the enum.Flag ``ColumnType``
    ``column_type``.

    TODO: Move this function away from here :)

    Args:
        df: Input DataFrame.
        column_type: ColumnType object, e.g., ``ColumnType.FLOAT``. They can be combined
         using the bitwise or operator ``|``, e.g.,
         ``ColumnType.FLOAT | ColumnType.STRING``.

    Returns:
        A list of strings which are the column names maching the given type(s) in
        ``column_type``
    """
    if ColumnType.ALL in column_type:
        return df.columns.to_list()

    dtypes = {i: str(i) for i in ColumnType if i != ColumnType.ALL}
    selection = []
    for column_type_single, type_name in dtypes.items():
        if column_type_single in column_type:
            local_selection = df.select_dtypes(include=type_name).columns.to_list()
            logger.info(f"{column_type_single} matches with {local_selection}.")
            selection.extend(local_selection)

    selection = list(set(selection))
    logger.info(f"Selected: {selection}.")

    ordering = {column: idx for idx, column in enumerate(df.columns)}

    selection = [(i, ordering[i]) for i in selection]
    selection = sorted(selection, key=lambda x: x[-1])
    selection = [i[0] for i in selection]
    logger.info(f"Reordered selection: {selection}.")
    return selection


@validate_call(config=_VALIDATE_CONFIG)
def _validate_y(
    y: Union[pd.DataFrame, pd.Series, None] = None
) -> Union[pd.DataFrame, None]:
    """Validate that ``y`` is a pd.DataFrame, pd.Series or None. If ``y``, cast
    it onto a pd.DataFrame."""
    if isinstance(y, pd.Series):
        y = y.to_frame(name="values" if y.name is None else y.name)

    return y


@validate_call(config=_VALIDATE_CONFIG)
def _validate_X(X: pd.DataFrame) -> pd.DataFrame:
    """Validate that ``X`` is a pd.DataFrame."""
    return X


@validate_call(config=_VALIDATE_CONFIG)
def _validate_X_y(
    X: Union[pd.DataFrame],
    y: Union[pd.DataFrame, pd.Series, None] = None,
    same_length: bool = True,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    """Validate ``X`` and ``y``.

    Args:
        X: Feature matrix to validate. It must be a pd.DataFrame.
        y: Target matrix/vector to validate. It must be a pd.Series, pd.DataFrame or
            None (in which case no validation is performed). If ``y`` is a pd.Series,
            it will be cast onto a single column pd.DataFrame. Defaults to None.
        same_length: If ``True`` and ``y`` is not None, ``X`` and ``y`` are checked
            to have the same length (otherwise, raise a ``ValueError``). Defaults to
            True.

    Returns:
        The validated tuple ``X, y``.
    """
    X = _validate_X(X)
    y = _validate_y(y)

    if same_length and y is not None and len(y) != len(X):
        raise ValueError(
            f"`X` and `y` should have the same length. Found {len(X)=} and {len(y)=}"
        )

    return X, y


class BaseMixin(ABC):
    def __init_subclass__(cls) -> None:
        _required_class_variables = ["_considered_types"]
        for cvar in _required_class_variables:
            if getattr(cls, cvar, UNSET) is UNSET:
                raise AttributeError(
                    f"ClassVar {cvar} need to be set for {cls.__name__}!"
                )

        return super().__init_subclass__()

    @validate_call(config=_VALIDATE_CONFIG)
    def _register_names(
        self,
        X: Union[pd.DataFrame],
        y: Union[pd.DataFrame, None] = None,
        if_name_exist: Literal["warn", "ignore"] = "warn",
    ) -> None:

        if (
            getattr(self, "feature_names_in_", UNSET) is not UNSET
            and if_name_exist == "warn"
        ):
            logger.warning(
                (
                    f"`feature_names_in_` is already set to {self.fitted_features_} "
                    f"in object {self.__class__.__name__}. You are probably calling "
                    f"`self.fit(...)` several times!"
                )
            )

        self.all_feature_names_in_ = X.columns.to_list()
        self.fitted_features_ = filter_columns_by_type(X, self._considered_types)

        # y might be None, e.g., transformers
        self.target_names_in_ = None
        if isinstance(y, pd.DataFrame):
            self.target_names_in_ = y.columns.to_list()

    @validate_call(config={"arbitrary_types_allowed": True})
    def _check_names(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, None] = None,
    ) -> None:

        missing = set(self.fitted_features_) - set(X.columns)
        if missing:
            raise KeyError(
                (
                    "Inconsistent names in ``X`` with respect to the last `fit` call! "
                    f"Columns {missing} were used during `fit` but are not in X! "
                    f"{X.columns=}."
                )
            )

        # TODO: is this needed? `fit` should be called once only anyway?
        if y is not None:
            missing = set(self.target_names_in_) - set(y.columns)
            if missing:
                raise KeyError(
                    (
                        "Inconsistent names in y with respect to the last `fit` call! "
                        f"Columns {missing} were seen during `fit` but are not in y! "
                        f"{y.columns=}."
                    )
                )

    def _has_fit(self) -> bool:
        return hasattr(super(), "fit")

    def _has_transform(self) -> bool:
        return hasattr(super(), "transform")

    def _has_inverse_transform(self) -> bool:
        return hasattr(super(), "inverse_transform")

    def _has_predict(self) -> bool:
        return hasattr(super(), "predict")

    #########################
    ### Sklearn extension ###
    #########################

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, None] = None, **kwargs
    ) -> Self:
        X, y = _validate_X_y(X, y, same_length=True)
        self._register_names(X, y, if_name_exist="warn")
        return super().fit(X[self.fitted_features_].copy(), y, **kwargs)

    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X, _ = _validate_X_y(X, None)
        self._check_names(X, None)

        X_new = X[self.fitted_features_].copy()
        X_new = super().transform(X=X_new, **kwargs)
        X_new = pd.DataFrame(X_new, columns=self.fitted_features_, index=X.index)

        missing = set(X.columns) - set(self.fitted_features_)
        for column in missing:
            X_new.loc[:, column] = X[column]

        # Re-order the columns if possible!
        if set(X_new.columns) == set(X.columns):
            return X_new[X.columns.to_list()]
        return X_new

    def inverse_transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X, _ = _validate_X_y(X, None)
        self._check_names(X, None)

        X_new = X[self.fitted_features_].copy()
        X_new = super().inverse_transform(X=X_new, **kwargs)
        X_new = pd.DataFrame(X_new, columns=self.fitted_features_, index=X.index)

        missing = set(X.columns) - set(self.fitted_features_)
        for column in missing:
            X_new.loc[:, column] = X[column]

        # Re-order the columns if possible!
        if set(X_new.columns) == set(X.columns):
            return X_new[X.columns.to_list()]
        return X_new

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        X, _ = _validate_X_y(X, None)
        self._check_names(X, None)

        X_new = X[self.fitted_features_].copy()
        y_hat = super().predict(X_new, **kwargs)

        y_hat = pd.DataFrame(y_hat, columns=self.target_names_in_, index=X.index)
        return y_hat
