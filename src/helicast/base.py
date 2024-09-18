from abc import ABC, abstractmethod
from enum import auto
from functools import cached_property, partial
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import pydantic.dataclasses
from pydantic import validate_call
from sklearn.base import BaseEstimator as _SKLearnBaseEstimator
from sklearn.base import check_is_fitted
from sklearn.metrics import r2_score
from strenum import StrEnum
from typing_extensions import Self, dataclass_transform

from helicast.utils import get_param_type_mapping, validate_equal_to_reference

__all__ = [
    "dataclass",
    "EstimatorMode",
    "validate_X",
    "validate_y",
    "validate_X_y",
    "HelicastBaseEstimator",
    "TransformerMixin",
    "StatelessTransformerMixin",
    "InvertibleTransformerMixin",
    "PredictorMixin",
]


DEFAULT_CONFIG = {"arbitrary_types_allowed": True}


@dataclass_transform()
def dataclass(cls=None, *, config=None):
    """Dataclass decorator for the Helicast library. It uses Pydantic under the hood,
    with minimal checks to ensure that nothing breaks. The main difference with the
    Pydantic dataclass is that the default config is set to allow arbitrary types.
    Also, the 'validate_assignment' cannot be set to True in the config (otherwise it
    would break on-the-fly assignements).
    """
    if cls is None:
        return partial(dataclass, config=config)

    if config is not None and config.get("validate_assignment", False) is True:
        raise RuntimeError(
            "Do not use 'validate_assignment=True' in the Pydantic config in child classes!"
        )
    if config is None:
        config = DEFAULT_CONFIG

    return pydantic.dataclasses.dataclass(_cls=cls, config=config)


class EstimatorMode(StrEnum):
    FIT = auto()
    TRANSFORM = auto()
    INVERSE_TRANSFORM = auto()
    PREDICT = auto()


@validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
def validate_X(obj: object, X: pd.DataFrame, *, mode: EstimatorMode) -> pd.DataFrame:
    # If fitting, store the feature names and return the input data
    if mode == EstimatorMode.FIT:
        obj.feature_names_in_ = X.columns.tolist()
        return X

    # If transforming or predicting, validate the input data
    if mode == EstimatorMode.TRANSFORM or mode == EstimatorMode.PREDICT:
        features = obj.feature_names_in_
    elif mode == EstimatorMode.INVERSE_TRANSFORM:
        features = obj.feature_names_out_
    else:
        raise RuntimeError(f"Unexpected mode: {mode=}")

    columns = validate_equal_to_reference(X.columns, features, reorder=True)
    return X[columns]


@validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
def validate_y(
    obj: object, y: Union[pd.DataFrame, pd.Series, None], *, mode: EstimatorMode
) -> Union[pd.DataFrame, None]:
    if y is None:
        return None
    elif mode == EstimatorMode.FIT:
        if isinstance(y, pd.Series) and y.name is None:
            raise ValueError(
                "y is a pd.Series with a None name. The name must be defined!"
            )
        if isinstance(y, pd.Series):
            y = y.to_frame(name=y.name)
        obj.target_names_in_ = y.columns.tolist()
    elif mode == EstimatorMode.PREDICT:
        if isinstance(y, pd.Series):
            if len(obj.target_names_in_) != 1:
                raise ValueError(
                    f"{len(obj.target_names_in_)=} is not 1 but we got y as a pd.Series."
                )
            y = y.to_frame(name=obj.target_names_in_[0])
        columns = validate_equal_to_reference(
            y.columns, obj.target_names_in_, reorder=True
        )
        y = y[columns]
    else:
        raise RuntimeError(f"Validating y in {mode=}. This is unexpected!")
    return y


def validate_X_y(
    obj: object,
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series, None] = None,
    *,
    mode: EstimatorMode,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    X = validate_X(obj, X, mode=mode)
    y = validate_y(obj, y, mode=mode)

    if mode == EstimatorMode.FIT and y is not None:
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Found {len(X)=} and {len(y)=}."
            )
        if not X.index.equals(y.index):
            raise ValueError(
                "Found different indices for X and y! They should be the same."
            )

    return X, y


@dataclass
class HelicastBaseEstimator(_SKLearnBaseEstimator, ABC):
    """Base class for all estimators in Helicast. It is a Pydantic dataclass that
    inherits from the sklearn BaseEstimator and adds some additional features. While
    sklearn estimators use numpy arrays as main data structure, Helicast estimators
    use pandas DataFrames. This is to ensure that the data is always properly
    labeled.

    !!! Do not use "validate_assignment" in the Pydantic config in child classes !!!
    """

    @cached_property
    def _param_names(self) -> List[str]:
        """Return a list of parameter names for the estimator, that is, the names of
        the attributes that are defined in __init__."""
        return self._get_param_names()

    def __setattr__(self, name: str, value: Any) -> None:
        """Override the __setattr__ method to ensure that the assignment is validated
        for the parameters that are defined in the __init__ method. For the other
        attributes, no check is performed."""
        if name in self._param_names:
            self.__pydantic_validator__.validate_assignment(self, name, value)
        else:
            super().__setattr__(name, value)

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        return super().__repr__(N_CHAR_MAX)

    ##################
    ### VALIDATION ###
    ##################

    def _validate_X(self, X: pd.DataFrame, *, mode: EstimatorMode) -> pd.DataFrame:
        return validate_X(self, X, mode=mode)

    def _validate_y(
        self, y: Union[pd.DataFrame, pd.Series, None], *, mode: EstimatorMode
    ) -> Union[pd.DataFrame, None]:
        return validate_y(self, y, mode=mode)

    def _validate_X_y(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, None] = None,
        *,
        mode: EstimatorMode,
    ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
        return validate_X_y(self, X, y, mode=mode)

    ###################
    ### SKLEARN API ###
    ###################

    @classmethod
    def _get_param_names(cls) -> List[str]:
        """Get parameter names for the estimator. This replace the original sklearn
        implementation to ensure full compatibility with Pydantic dataclasses."""
        return list(get_param_type_mapping(cls).keys())

    @classmethod
    def _get_param_types(cls) -> Dict[str, type]:
        """Returns the types of the parameters of the estimator. This is useful for
        type checking."""
        return get_param_type_mapping(cls)

    @abstractmethod
    def _fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> Self:
        """Internal function to fit the estimator. This function should not validate
        the input data, as this is done in the fit method. This function should return
        the fitted estimator."""
        raise NotImplementedError()

    @validate_call(config={"arbitrary_types_allowed": True})
    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, None] = None, **kwargs
    ) -> Self:
        """Fit the estimator to the data.

        Args:
            X: Features. Shape (n_samples, n_features).
            y: Target. Can be None if the estimator is unsupervised (e.g., min-max
                scaler). If not Nonte, the shape should be (n_samples, n_targets) or
                (n_samples,). Defaults to None.

        Returns:
            The fitted estimator.
        """
        X, y = self._validate_X_y(X, y, mode=EstimatorMode.FIT)
        self._fit(X, y, **kwargs)

        # Try to force a transformation to ensure that the feature names
        # are stored for invertible transformers
        try:
            self.transform(X.iloc[: min(1, len(X))])
        except Exception:
            pass

        return self


class TransformerMixin(ABC):
    """Mixin class for non-invertible statefull transformers in Helicast.
    Classes that inherit from this mixin should implement the ``_transform`` method and
    the ``_fit`` method."""

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = self._validate_X(X, mode=EstimatorMode.TRANSFORM)
        X_tr = self._transform(X)

        if not isinstance(X_tr, pd.DataFrame):
            raise ValueError(f"Expected a pd.DataFrame but got {type(X_tr)} instead.")

        self.feature_names_out_ = X_tr.columns.tolist()
        return X_tr

    def fit_transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, None] = None, **kwargs
    ) -> pd.DataFrame:
        self.fit(X, y, **kwargs)
        return self.transform(X)


class StatelessTransformerMixin(TransformerMixin):
    """Mixin class for non-invertible stateless transformers in Helicast.
    Classes that inherit from this mixin should implement the ``_transform``.
    The ``_fit`` method has a default implementation that does nothing."""

    def _fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> Self:
        return self


class InvertibleTransformerMixin(TransformerMixin):
    """Mixin class for invertible statefull transformers in Helicast.
    Classes that inherit from this mixin should implement the ``_fit``, ``_transform``
    and ``_inverse_transform`` method."""

    @abstractmethod
    def _inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = self._validate_X(X, mode=EstimatorMode.INVERSE_TRANSFORM)
        X_inv_tr = self._inverse_transform(X)
        X_inv_tr = self._validate_X(X_inv_tr, mode=EstimatorMode.TRANSFORM)
        return X_inv_tr


class PredictorMixin(ABC):
    """Mixin class for predictors (e.g., linear model) in Helicast.
    Classes that inherit from this mixin should implement the ``_fit`` and ``_predict``
    methods."""

    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the input data. The estimator must be fitted
        before calling this method.

        Contrarily to the sklearn API, the input data should always be a DataFrame and
        the output is always a DataFrame (even for regression tasks with only one
        target variable). This is to ensure that the data is always properly labeled.

        Args:
            X: Features. Shape (n_samples, n_features).

        Returns:
            The predicted target values. Shape (n_samples, n_targets).
        """
        check_is_fitted(self)
        X = self._validate_X(X, mode=EstimatorMode.PREDICT)
        y_pred = self._predict(X)
        y_pred = self._validate_y(y_pred, mode=EstimatorMode.PREDICT)
        return y_pred

    def fit_predict(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, None] = None, **kwargs
    ) -> pd.DataFrame:
        """Syntactic sugar to fit the estimator and predict the target values in one
        step. Refer to the ``fit`` and ``predict`` methods for more information.

        Args:
            X: Features. Shape (n_samples, n_features).
            y: Target. Can be None if the estimator is unsupervised, otherwise it
                should have shape (n_samples, n_targets) or (n_samples,). Defaults to
                None.

        Returns:
            The predicted target values. Shape (n_samples, n_targets).
        """
        self.fit(X, y, **kwargs)
        return self.predict(X)

    def score(self, X, y, sample_weight=None) -> float:
        y_pred = self.predict(X)
        score = r2_score(y, y_pred, sample_weight=sample_weight)
        return float(score)
