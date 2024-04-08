import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import wraps
from typing import Any, ClassVar, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    validate_call,
)
from sklearn.base import BaseEstimator
from sklearn.exceptions import DataConversionWarning
from typing_extensions import Self

from helicast.logging import configure_logging
from helicast.typing import UNSET, UnsetType

configure_logging()
logger = logging.getLogger(__name__)


_VALIDATE_CONFIG = {
    "arbitrary_types_allowed": True,
    "validate_return": True,
    "strict": True,
}

__all__ = [
    "PydanticBaseEstimator",
    "DFTransformer",
    "StatelessDFTransformer",
]


@validate_call(config=_VALIDATE_CONFIG)
def reoder_columns_if_needed(
    df: pd.DataFrame,
    columns: Union[pd.Index, np.ndarray, List[str]],
    strict: bool = True,
) -> pd.DataFrame:
    if strict:
        missing = set(columns) - set(df.columns)
        extra = set(df.columns) - set(columns)
        if missing or extra:
            raise IndexError(
                (
                    f"Trying to reorder in strict mode but columns {sorted(missing)} "
                    f"are missing and columns {sorted(extra)} are not valid!"
                )
            )

    if list(df.columns) == list(columns):
        return df
    else:
        new_ordering = [i for i in columns if i in df.columns]
        return df[new_ordering]


@validate_call(config=_VALIDATE_CONFIG)
def cast_to_DataFrame(
    array: Union[pd.DataFrame, pd.Series, np.ndarray],
    columns: Union[pd.Index, List[Any], np.ndarray],
    index: Union[pd.Index, List[Any], np.ndarray],
    allowed_ndims: Tuple[int, ...] = (1, 2),
) -> pd.DataFrame:
    if isinstance(array, pd.Series):
        array = array.to_numpy()

    if isinstance(array, pd.DataFrame):
        array = reoder_columns_if_needed(array, columns)

        if not array.index.equals(index):
            raise RuntimeError(f"{array.index=}, {index=}")
    elif isinstance(array, np.ndarray):

        if array.ndim not in set(allowed_ndims):
            raise ValueError(f"{array.ndim=}, {allowed_ndims=}")

        if array.ndim == 1:
            array = np.reshape(array, (-1, 1))

        array = pd.DataFrame(array, columns=columns, index=index)

    return array


def predict(self, X: pd.DataFrame, **predict_params) -> pd.DataFrame:
    _check_fitted(self, ["_feature_names_in", "_target_names_in"])
    X = _validate_X(X)
    _check_X_columns(self, X, self._feature_names_in)

    # Reorder columns
    y_hat = self.regressor.predict(X[self._feature_names_in], **predict_params)

    # Cast onto a pd.DataFrame is needed
    if isinstance(y_hat, pd.DataFrame):
        pass
    elif isinstance(y_hat, (np.ndarray, pd.Series)):
        if isinstance(y_hat, pd.Series):
            y_hat = y_hat.to_numpy()

        if y_hat.ndim == 1:
            y_hat = np.reshape(y_hat, (-1, 1))

        y_hat = pd.DataFrame(y_hat, columns=self._target_names_in, index=X.index)
    else:
        raise TypeError(f"{type(y_hat)=}")

    # Inverse
    y_hat = self.transformer.inverse_transform(y_hat)

    if isinstance(y_hat, pd.DataFrame):
        if set(y_hat.columns) != set(self._target_names_in):
            raise ValueError(f"{y_hat.columns=}, {self._target_names_in=}.")
    elif isinstance(y_hat, pd.Series):
        if len(self._target_names_in) != 1:
            raise ValueError(f"{self._target_names_in=}")
        y_hat = y_hat.to_frame(
            name=y_hat.name if y_hat.name is not None else self._target_names_in[0]
        )
    elif isinstance(y_hat, np.ndarray):
        if y_hat.ndim == 1:
            y_hat = np.reshape(y_hat, (-1, 1))
        y_hat = pd.DataFrame(y_hat, columns=self._target_names_in, index=X.index)
    else:
        raise TypeError(f"Unexpected type {type(y_hat)=}")

    return y_hat


def ignore_data_conversion_warnings(func):
    """Decorator to ignore ``DataConversionWarning`` from sklearn (some estimators
    expect a 1D array as ``y``, while we work mostly with DataFrame)."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DataConversionWarning)
            result = func(*args, **kwargs)
        return result

    return wrapper


####################
# Helper Functions #
####################
def _check_fitted(obj, attrs: Union[List[str], None] = None):
    """Raise a `RuntimeError` if ``obj`` is not fitted. ``obj`` is considerd not fitted
    if any of the attributes specified in ``attrs`` is ``UNSET``.

    If ``attrs`` is None (default), it is set to
    ``["feature_names_in_", "target_names_in_"]``."""
    if attrs is None:
        attrs = ["feature_names_in_", "target_names_in_"]

    if any(getattr(obj, i, UNSET) is UNSET for i in attrs):
        raise RuntimeError(f"Transformer {obj.__class__.__name__} is not fitted!")


def _check_X_columns(
    obj,
    X: pd.DataFrame,
    feature_names_in: Union[List[str], None] = None,
):
    """Raise a ``RuntimeError`` if the columns of ``X`` do not correspond to the columns
    seen by ``obj`` during fit which are (normally) stored in the
    ``obj.feature_names_in_`` attribute. If ``obj`` has no attribute
    ``feature_names_in_``, the function argument ``feature_names_in`` will be used,
    otherwise, the function argument ``feature_names_in`` is ignored."""

    if hasattr(obj, "feature_names_in_"):
        feature_names_in = deepcopy(getattr(obj, "feature_names_in_"))
    if feature_names_in is None or feature_names_in is UNSET:
        raise TypeError(f"{feature_names_in=}")

    columns = X.columns.to_list()

    extra = set(columns) - set(feature_names_in)
    missing = set(feature_names_in) - set(columns)
    msg = []
    if extra:
        msg.append(f"Columns {sorted(extra)} were not seen during fit!")
    if missing:
        msg.append(
            f"Columns {sorted(missing)} were seen during fit but are now missing!"
        )
    if msg:
        msg = "\n".join([f" - {i}" for i in msg])
        raise RuntimeError(f"The following errors were found for X:\n{msg}")


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
    allow_y_None: bool = True,
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

    if not allow_y_None and y is None:
        raise TypeError("`y` is NOT allowed to be None!")

    if same_length and y is not None and len(y) != len(X):
        raise ValueError(
            f"`X` and `y` should have the same length. Found {len(X)=} and {len(y)=}"
        )

    return X, y


class PydanticBaseEstimator(BaseEstimator, BaseModel):
    """Extend the BaseEstimator from sklearn to include some Pydantic magic :)"""

    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, "model_config"):
            cls.model_config = ConfigDict()

        forced_defaults = {
            "arbitrary_types_allowed": True,
            "validate_assignment": True,
        }

        for k, v in forced_defaults.items():
            if k in cls.model_config.keys() and cls.model_config[k] != v:
                logger.warning(
                    (
                        f"{cls.__name__}: replacing {k}={cls.model_config[k]} in "
                        f"model_config to {k}={v}. This is necessary to ensure a proper "
                        f"behavior of PydanticBaseEstimator subclasses!"
                    )
                )
            cls.model_config[k] = v

        return super().__init_subclass__(**kwargs)

    @classmethod
    def _get_param_names(cls) -> List[str]:
        """Overriding sklearn BaseEstimator ``_get_param_names`` classmethod to
        accomodate for Pydantic BaseModel."""
        args = cls.model_fields
        args = {i: j for i, j in args.items() if j.init is not False}

        return list(sorted(args.keys()))

    def __str__(self) -> str:
        args = ", ".join(
            [f"{i}={str(getattr(self, i))}" for i in self.model_fields.keys()]
        )
        name = self.__class__.__name__
        return f"{name}({args})"


class FitMixin:
    """Base class for object that can be fitted as a sklearn-like object."""

    ###################
    # Class Variables #
    ###################
    # Define the Class variables that control the behavior of the class. Modify their
    # values in the child classes to control the behavior

    #: If True, the names during `self.transform(...)` must match the one seen during `self.fit(...)`
    _strict_feature_names_in: ClassVar[bool] = True

    #: If Ture, `self.transform(...)` cannot be called without prior call to `self.fit(...)`
    _requires_fit: ClassVar[bool] = True

    ######################
    # Private Attributes #
    ######################
    # Private attributes, used with properties (for pydantic/sphinx interaction)

    _feature_names_in: List[str] | UnsetType = PrivateAttr(UNSET)
    _target_names_in: List[str] | UnsetType | None = PrivateAttr(UNSET)

    @property
    def feature_names_in_(self) -> List[str] | UnsetType:
        return deepcopy(self._feature_names_in)

    @feature_names_in_.setter
    def _(self, value):
        value = TypeAdapter(List[str], config={"strict": True})
        self._feature_names_in = value[:]

    @property
    def target_names_in_(self) -> List[str] | UnsetType:
        return deepcopy(self._target_names_in)

    @target_names_in_.setter
    def _(self, value: List[str] | None):
        if value is None:
            self._target_names_in = None
        else:
            value = TypeAdapter(List[str], config={"strict": True})
            self._target_names_in = value[:]

    ##################
    # Helper Methods #
    ##################
    def _check_fitted(self):
        """Raise a `RuntimeError` if the transformer is not fitted, that is, if
        self.feature_names_in_ or self.target_names_in_ are UNSET.

        If the ClassVar `_requires_fit` is `False`, this method never raise!
        """
        if not self._requires_fit:
            return None

        attrs = ["feature_names_in_", "target_names_in_"]

        if any(getattr(self, i) is UNSET for i in attrs):
            raise RuntimeError(f"Transformer {self.__class__.__name__} is not fitted!")

    def _save_names(self, X: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """Save the column names of `X` and optionally `y` (if not None)."""
        self.feature_names_in_ = X.columns.to_list()
        self.target_names_in_ = None
        if y is not None:
            self.target_names_in_ = y.columns.to_list()

    @validate_call(config=_VALIDATE_CONFIG)
    def _check_X_names(self, X: pd.DataFrame):
        """Raise a ``RuntimeError`` if the the columns in ``X`` are not the same
        as the one seen during ``self.fit(...)`` and the ClassVar
        ``self._strict_feature_names_in`` is ``True``. Otherwise, returns silently
        ``None``.

        This method should be called at the beginning of methods that requires the
        object to be fitted first, e.g., ``self.transform(...)``, ``self.predict(...)``,
        ..."""

        if not self._strict_feature_names_in:
            return None

        self._check_fitted()

        columns_transform = X.columns.to_list()
        columns_fit = self.feature_names_in_

        extra = set(columns_transform) - set(columns_fit)
        missing = set(columns_fit) - set(columns_transform)
        msg = []
        if extra:
            msg.append(f"Columns {sorted(extra)} were not seen during fit!")
        if missing:
            msg.append(
                f"Columns {sorted(missing)} were seen during fit but are now missing!"
            )
        if msg:
            msg = "\n".join([f" - {i}" for i in msg])
            raise RuntimeError(f"The following errors were found for X:\n{msg}")

    ####################
    # Scikit-Learn API #
    ####################

    @abstractmethod
    def _fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, None] = None,
        **fit_params,
    ) -> Self:
        """Abstract method to be implemented in child classes."""
        pass

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, None] = None,
        **fit_params,
    ) -> Self:
        f"""Fit the {self.__class__.__name__} object. The input feature DataFrame ``X``
        and, optionally, the target DataFrame/Series ``y`` is type-validated.

        Args:
            X: Input DataFrame.
            y: Optional target DataFrame/Series for supervised transformers. Defaults to
             None.

        Returns:
            The object instance that is fitted (self).
        """
        X, y = _validate_X_y(X, y)
        self._save_names(X, y)

        self._fit(X, y, **fit_params)
        return self


class TransformerMixin:
    @abstractmethod
    def _transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, None] = None,
    ) -> pd.DataFrame:
        pass

    def transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, None] = None,
    ) -> pd.DataFrame:
        f"""Transform the input DataFrame ``X``. If the object requires a call to fit
        but has not been fitted, a RuntimeError will be raised. 

        Args:
            X: Input DataFrame.
            y: Optional target DataFrame/Series for supervised transformers. Defaults to
             None.

        Returns:
            The transformed DataFrame.
        """

        self._check_fitted()
        X, y = _validate_X_y(X, y)
        self._check_X_names(X)

        if set(X.columns.to_list()) == set(self.feature_names_in_):
            X = X[self.feature_names_in_]

        return self._transform(X, y)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, None] = None,
        **fit_params,
    ) -> pd.DataFrame:
        f"""Fit the {self.__class__.__name__} object and use the fitted object to 
        transform the input DataFrame ``X``. This is equivalent to calling
        ``{self.__class__.__name__}.fit(X, y, **fit_params).transform(X, y)``.

        Args:
            X: Input DataFrame.
            y: Optional target DataFrame/Series for supervised transformers. Defaults to
             None.

        Returns:
            The transformed DataFrame.
        """

        return self.fit(X, y, **fit_params).transform(X, y)


class BaseRegressor:
    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        f"""Predict using the {self.__class__.__name__} object.
        
        * If the columns in ``X`` do not match with the ones seen during
         ``self.fit(...)``, raise ``RuntimeError``.
        * If the object has not been fitted, raise ``RuntimeError``.

        Args:
            X: Input DataFrame.

        Returns:
            The prediction as a DataFrame.
        """

        self._check_fitted()
        X, y = _validate_X_y(X, y)
        self._check_X_names(X)

        if set(X.columns.to_list()) == set(self.feature_names_in_):
            X = X.copy()[self.feature_names_in_]
        else:
            X = X.copy()

        y_hat = self._predict(X)

        if isinstance(y_hat, pd.Series):
            y_hat = y_hat.to_numpy()

        if isinstance(y_hat, pd.DataFrame):
            if set(y_hat.columns) != set(self.target_names_in_):
                raise KeyError(f"{y_hat.columns=}, {self.target_names_in_=}")
        elif isinstance(y_hat, np.ndarray):
            if y_hat.ndim == 1:
                y_hat = np.reshape(y_hat, (-1, 1))

            if y_hat.ndim != 2:
                raise ValueError(f"{y_hat.ndim=}")
            if y_hat.shape[-1] != len(self.target_names_in_):
                raise ValueError(f"{y_hat.shape=}, {self.target_names_in_=}")

            y_hat = pd.DataFrame(y_hat, columns=self.target_names_in_, index=X.index)
        else:
            raise TypeError(f"{type(y)=}")

        return y_hat


class DFTransformer(PydanticBaseEstimator, ABC):
    """Abstract base class for transformers that are stefull, i.e., that cannot call
    ``self.transform`` without a prior ``self.fit`` call.

    You need to implement the abstract methods

    .. code-block::python

        @abstractmethod
        def _fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, None], **kwargs) -> None:
            pass

        def _transform(
            self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
        ) -> pd.DataFrame:
            pass
    """

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "is_fitted_", False)

    @abstractmethod
    def _fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        pass

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> Self:
        X, y = _validate_X_y(X, y, same_length=True)
        self._fit(X, y, **kwargs)
        self.is_fitted_ = True
        return self

    def transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Not fitted!")

        X, y = _validate_X_y(X, y, same_length=True)
        X_new = self._transform(X, y, **kwargs)
        if not isinstance(X_new, pd.DataFrame):
            raise TypeError(f"{type(X_new)=} should be pd.DataFrame!")
        return X_new

    def fit_transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        X, y = _validate_X_y(X, y, same_length=True)
        return self.fit(X, y, **kwargs).transform(X, y, **kwargs)


class StatelessDFTransformer(DFTransformer):
    """Abstract base class for transformers that are stateless, i.e., that can call
    ``self.transform`` without the need of a ``self.fit`` prior call.

    You need to implement the abstract method

    .. code-block::python

        def _transform(
            self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
        ) -> pd.DataFrame:
            pass
    """

    @property
    def is_fitted(self) -> bool:
        return True

    def _fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> None:
        pass

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> Self:
        return self
