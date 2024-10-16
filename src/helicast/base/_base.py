from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property
from typing import Any
from warnings import warn

import numpy as np
import pandas as pd
from pydantic import validate_call
from pydantic.dataclasses import is_pydantic_dataclass
from sklearn.base import BaseEstimator as _SKLearnBaseEstimator
from sklearn.base import clone
from typing_extensions import Self

from helicast._version import __version__
from helicast.base._dataclass import dataclass
from helicast.typing import EstimatorMode
from helicast.utils import (
    get_param_type_mapping,
    numpy_array_to_dataframe,
    series_to_dataframe,
    validate_column_names_as_string,
    validate_equal_to_reference,
    validate_subset_of_reference,
)

__all__ = [
    "HelicastBaseEstimator",
    "StatelessEstimator",
    "validate_X",
    "validate_X_y",
    "validate_y",
    "is_stateless",
]


@validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
def validate_y(
    obj: object,
    y: pd.DataFrame | pd.Series | np.ndarray | None,
    *,
    mode: EstimatorMode,
    index: pd.Index | None = None,
) -> pd.DataFrame | None:
    if y is None:
        return None

    match mode:
        case EstimatorMode.FIT:
            if isinstance(y, pd.Series):
                y = series_to_dataframe(y)
            obj.target_names_in_ = y.columns.tolist()
            return y

        case EstimatorMode.PREDICT:
            if isinstance(y, np.ndarray):
                if index is None:
                    raise ValueError("y is a np.ndarray, the index cannot be None!")
                y = numpy_array_to_dataframe(y, obj.target_names_in_, index)
            elif isinstance(y, pd.Series):
                if len(obj.target_names_in_) != 1:
                    raise ValueError(
                        f"{obj.target_names_in_=} has more than 1 element but we "
                        f"got y as a pd.Series in predict mode."
                    )
                y = series_to_dataframe(y, name=obj.target_names_in_[0])

        case _:
            raise RuntimeError(f"Validating y in {mode=}. This is unexpected!")

    columns = validate_equal_to_reference(y.columns, obj.target_names_in_, reorder=True)
    return y[columns]


@validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
def validate_X(obj: object, X: pd.DataFrame, *, mode: EstimatorMode) -> pd.DataFrame:
    X = validate_column_names_as_string(X)

    # Stateless classes do not require the full column names validation
    if is_stateless(obj):
        obj.feature_names_in_ = X.columns.tolist()
        return X

    match mode:
        case EstimatorMode.FIT:
            obj.feature_names_in_ = X.columns.tolist()
            return X

        case EstimatorMode.TRANSFORM | EstimatorMode.PREDICT:
            columns = list(obj.feature_names_in_)

        case EstimatorMode.INVERSE_TRANSFORM:
            columns = list(obj.feature_names_out_)

        case _:
            raise RuntimeError(f"Unexpected mode: {mode=}")

    # Set ``_disable_X_column_check = True`` to disable this check
    if getattr(obj, "_disable_X_column_subset_filtering", False) is False:
        validate_subset_of_reference(columns, X.columns)
        return X[columns]
    else:
        return X


def validate_X_y(
    obj: object,
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series | np.ndarray | None = None,
    index: pd.Index | None = None,
    *,
    mode: EstimatorMode,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    X = validate_X(obj, X, mode=mode)
    y = validate_y(obj, y, mode=mode, index=index)

    if mode == EstimatorMode.FIT and y is not None:
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Found {len(X)=} and {len(y)=}."
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
    def _param_names(self) -> list[str]:
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
            object.__setattr__(self, name, value)
            # super().__setattr__(name, value)

    ###################
    ### SKLEARN API ###
    ###################
    def __sklearn_is_fitted__(self) -> bool:
        name = "feature_names_in_"
        return hasattr(self, name) and isinstance(getattr(self, name), list)

    def __sklearn_clone__(self) -> Self:
        params = {
            k: clone(v, safe=False) for k, v in self.get_params(deep=False).items()
        }
        return self.__class__(**params)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        params = {k: deepcopy(v, memo) for k, v in self.get_params(deep=False).items()}
        obj = self.__class__(**params)

        for k, v in self.__dict__.items():
            if k not in params.keys():
                setattr(obj, k, deepcopy(v, memo))

        return obj

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Get parameter names for the estimator. This replace the original sklearn
        implementation to ensure full compatibility with Pydantic dataclasses."""
        if is_pydantic_dataclass(cls):
            return list(get_param_type_mapping(cls).keys())
        else:
            return super()._get_param_names()

    @classmethod
    def _get_param_types(cls) -> dict[str, type]:
        """Returns the types of the parameters of the estimator. This is useful for
        type checking."""
        return get_param_type_mapping(cls)

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None, **kwargs) -> Self:
        """Internal function to fit the estimator. This function should not validate
        the input data, as this is done in the fit method. This function should return
        the fitted estimator."""
        raise NotImplementedError()

    @validate_call(config={"arbitrary_types_allowed": True})
    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None, **kwargs
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
        X, y = validate_X_y(self, X=X, y=y, mode=EstimatorMode.FIT)
        self._fit(X, y, **kwargs)

        # Transform a sample to get the output feature names for invertible transformers
        try:
            self.transform(X.iloc[: min(1, len(X))])
        except Exception:
            pass

        return self

    def __getstate__(self) -> dict:
        """Get the state of the object for pickling. This method is overridden to add
        the Helicast version to the state. This is useful to check for version
        consistency when unpickling the object."""

        state = super().__getstate__()

        if type(self).__module__.startswith("helicast."):
            return dict(state.items(), __helicast_version__=__version__)
        else:
            return state

    def __setstate__(self, state: dict) -> None:
        """Set the state of the object after pickling. This method is overridden to
        check for version consistency when unpickling the object. If the version of the
        object is different from the current version of Helicast, a warning is raised."""

        pickle_version = state.pop("__helicast_version__", None)
        if pickle_version is not None and pickle_version != __version__:
            warn(
                f"Inconsistent Helicast versions: the object was pickled with helicast "
                f"version {pickle_version} while you are currently using version "
                f"{__version__}. This is not recommended."
            )

        return super().__setstate__(state)


@dataclass
class StatelessEstimator(HelicastBaseEstimator):
    def __sklearn_is_fitted__(self) -> bool:
        return True

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None, **kwargs) -> Self:
        # Staless estimator do not require fitting
        return self

    @validate_call(config={"arbitrary_types_allowed": True})
    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None, **kwargs
    ) -> Self:
        """Fit method used to ensure compatibility with the API. This class is
        stateless so this method does nothing."""
        return self


def _get_class_name(obj: type | object) -> str:
    try:
        if isinstance(obj, type):
            return obj.__name__
        return obj.__class__.__name__
    except AttributeError:
        return str(obj)


def is_stateless(obj: type | object) -> bool:
    """Returns ``True`` if the ``obj`` is an instance of a stateless class or a
    stateless class, ``False`` otherwise.

    A class is considered stateless if it has a ``__helicast_is_stateless__`` attribute
    that is either (1) a callable that returns boolean or (2) a boolean, **OR** if the
    class is a subclass of ``StatelessEstimator``. The attribute
    ``__helicast_is_stateless__`` is checked first.

    If ``obj`` has a ``__helicast_is_stateless__`` attribute that is neither a boolean
    nor a callable that returns a boolean, a ``TypeError`` is raised."""

    class_name = _get_class_name(obj)

    if hasattr(obj, "__helicast_is_stateless__"):
        if callable(obj.__helicast_is_stateless__):
            if isinstance(obj, type):
                result = obj.__helicast_is_stateless__(None)
            else:
                result = obj.__helicast_is_stateless__()
        else:
            result = obj.__helicast_is_stateless__

        if not isinstance(result, bool):
            raise TypeError(
                f"Class {class_name}.__helicast_is_stateless__ must either be a bool "
                f"or a callable that returns a bool. Found {result=} ({type(result)=})."
            )
        return result

    if isinstance(obj, type):
        return issubclass(obj, StatelessEstimator)
    else:
        return isinstance(obj, StatelessEstimator)
