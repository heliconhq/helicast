from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from enum import Flag, auto
from logging import getLogger
from typing import ClassVar, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self, Unpack

from helicast.logging import configure_logging
from helicast.typing import UNSET
from helicast.validation import HelicastBaseModel
from helicast.yaml import get_init_args, register_yaml

configure_logging()
logger = getLogger(__name__)


__all__ = [
    "ColumnType",
    "filter_columns_by_type",
    "DataFrameEstimator",
    "StatelessDataFrameTransformer",
    "DataFrameTransformer",
]


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
            selection.extend(df.select_dtypes(include=type_name).columns.to_list())

    selection = list(set(selection))

    ordering = {column: idx for idx, column in enumerate(df.columns)}

    selection = [(i, ordering[i]) for i in selection]
    selection = sorted(selection, key=lambda x: x[-1])
    selection = [i[0] for i in selection]
    return selection


class DataFrameEstimator(BaseModel, ABC):
    """Base class imitating the sklearn-like interface but with a focus on DataFrame.
    The class itself does not depends on sklearn (no import needed)."""

    _considered_types: ClassVar[ColumnType] = UNSET

    _name_conserving: ClassVar[bool] = UNSET

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = "forbid"

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]):
        """Initialize each subclass of ``BaseEstimator`` to be compatible with YAML."""
        register_yaml(cls)
        return super().__init_subclass__(**kwargs)

    ##################
    ### VALIDATION ###
    ##################

    def _validate_X_y(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None
    ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
        """Validate ``X`` and optionally ``y`` (if not None). ``X`` must be a pandas
        DataFrame. For ``y``:

        * If ``y`` is None, nothing is done and ``y`` is returned;
        * If ``y`` is a pandas DataFrame, nothing is done and ``y`` is returned;
        * If ``y`` is a pandas Series, it is cast onto a pandas DataFrame (analogous
        to ``np.reshape(a, shape=(-1,1))``);
        * In any other case, a ``TypeError`` is raised.

        Returns:
            The validated ``X`` and ``y`` as a tuple. If the provided ``y`` was None,
            the return value type is ``Tuple[pd.DataFrame, None]``, otherwise,
            ``Tuple[pd.DataFrame, pd.DataFrame]``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X should be a pd.DataFrame. Found: {type(X)=}.")

        if isinstance(y, pd.Series):
            y = y.to_frame(name="values" if y.name is None else y.name)

        if (y is not None) and (not isinstance(y, pd.DataFrame)):
            raise TypeError(
                f"y should be of type pd.DataFrame, pd.Series or None. Found: {type(y)=}."
            )

        return X, y

    def _register_names(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
        if_name_exist: Literal["warn", "ignore"] = "warn",
    ) -> None:
        """Register the names of the columns of ``X`` (and ``y`` if not None). If the
        class has a ``_select_valid_columns`` method, the required columns will be
        filtered by that method."""

        if self.feature_names_in_ is not UNSET and if_name_exist == "warn":
            logger.warning(
                (
                    f"`feature_names_in_` is already set to {self.feature_names_in_} "
                    f"in object {self.__class__.__name__}. You are probably calling "
                    f"`self.fit(...)` several times!"
                )
            )

        self.feature_names_in_ = X.columns.to_list()
        self.all_feature_names_in_ = deepcopy(self.feature_names_in_)

        if self._considered_types is UNSET:
            raise RuntimeError(
                (
                    f"The ClassVar {self._considered_types=} is UNSET. "
                    "It should be defined in the source code! "
                )
            )
        self.feature_names_in_ = filter_columns_by_type(X, self._considered_types)

        if isinstance(y, pd.DataFrame):
            self.target_names_in_ = y.columns.to_list()
        elif y is None:
            self.target_names_in_ = None
        else:
            raise TypeError(
                f"y should be of type pd.DataFrame or None. Found: {type(y)=}."
            )

    def _check_names(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> None:

        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise KeyError(
                (
                    "Inconsistent names in X with respect to the last `fit` call! "
                    f"Columns {missing} were seen during `fit` but are not in X!"
                )
            )

        # TODO: is this needed? `fit` should be called once only anyway?
        if y is not None:
            missing = set(self.target_names_in_) - set(y.columns)
            if missing:
                raise KeyError(
                    (
                        "Inconsistent names in y with respect to the last `fit` call! "
                        f"Columns {missing} were seen during `fit` but are not in y!"
                    )
                )

    def _check_is_fitted(self) -> None:
        """Check if the object is fitted by looking at the ``self.feature_names_in_``
        attribute. It it is not fitted, raise a ``RuntimeError``."""
        if self.feature_names_in_ is UNSET:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted!")

    def is_name_conserving(self) -> bool:
        if self._name_conserving is UNSET:
            raise RuntimeError(f"{self._name_conserving=} should be set in the code!")
        return self._name_conserving

    def is_invertible(self) -> bool:
        return hasattr(self, "inverse_transform")

    ###########################
    ### SKLEARN DUCKTYPING  ###
    ###########################

    @classmethod
    def _get_param_names(cls):
        return sorted(get_init_args(cls))

    def get_params(self, deep: bool = True):
        """Taken from sklearn."""
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Taken from sklearn."""
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __str__(self) -> str:
        attributes = ", ".join(
            [f"{str(i)}={str(j)}" for i, j in self.get_params(deep=False).items()]
        )
        name = self.__class__.__name__
        return f"{name}({attributes})"

    def _append_missing_columns(
        self, X_old: pd.DataFrame, X_new: pd.DataFrame
    ) -> pd.DataFrame:

        missing_columns = set(X_old.columns) - set(self.feature_names_in_)
        for c in missing_columns:
            X_new.loc[:, c] = X_old[c]
        try:
            # Might not work if the transformer is not name conserving!
            X_new = X_new[[X_old.columns]]
        except:
            pass
        return X_new


class DataFrameTransformer(DataFrameEstimator):
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> None:
        pass

    @abstractmethod
    def _transform(
        self, X: pd.DataFrame, y: pd.DataFrame | None = None
    ) -> pd.DataFrame | np.ndarray:
        pass

    @abstractmethod
    def _inverse_transform(
        self, X: pd.DataFrame, y: pd.DataFrame | None = None
    ) -> pd.DataFrame | np.ndarray:
        pass

    ###########################
    ### SKLEARN INTERFACE  ###
    ###########################

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None):
        X, y = self._validate_X_y(X, y)
        self._register_names(X, y)
        self._fit(X[self.feature_names_in_], y)
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None
    ) -> pd.DataFrame:

        X, y = self._validate_X_y(X, y)
        self._check_is_fitted()
        self._check_names(X, y)

        index = X.index.copy()
        X_new = self._transform(X[self.feature_names_in_].copy(), y)

        if self.is_name_conserving():
            self.feature_names_out_ = deepcopy(self.feature_names_in_)
        else:
            if isinstance(X_new, np.ndarray):
                n_cols = X_new.shape[-1]
                self.feature_names_out_ = [f"column[{i}]" for i in range(n_cols)]
            elif isinstance(X_new, pd.DataFrame):
                self.feature_names_out_ = X_new.columns.to_list()
            else:
                raise TypeError(f"{type(X_new)=}")

        if isinstance(X_new, np.ndarray):
            X_new = pd.DataFrame(X_new, columns=self.feature_names_out_, index=index)
        elif isinstance(X_new, pd.DataFrame):
            pass
        else:
            raise TypeError(f"{type(X_new)=}")

        X_new = self._append_missing_columns(X, X_new)

        return X_new

    def fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X, y)

    def inverse_transform(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None
    ) -> pd.DataFrame:
        return self._transform_or_inverse_transform("inverse_transform", X, y)


class StatelessDataFrameTransformer(DataFrameTransformer):
    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> None:
        """A stateless transformer doesn't care about fitting!"""
        pass

    def _check_is_fitted(self) -> None:
        pass

    def _check_names(self, *args, **kwargs) -> None:
        pass

    ###########################
    ### SKLEARN INTERFACE  ###
    ###########################
    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None) -> Self:
        """A stateless transformer doesn't need to be fitted. This is a dummy method
        to ensure API compatibility."""
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None
    ) -> pd.DataFrame:
        """Takes a pd.DataFrame ``X`` and returns a transformed pd.DataFrame.

        Args:
            X: Input DataFrame
            y: Input target DataFrame or Series. Ignored if None. Defaults to None.
        """
        X, y = self._validate_X_y(X, y)
        self._register_names(X, y, if_name_exist="ignore")
        return super().transform(X, y)


class NotInvertibleMixin:
    def _inverse_transform(self, *args, **kwargs) -> None:
        name = self.__class__.__name__
        raise RuntimeError(f"{name} is not invertible!")
