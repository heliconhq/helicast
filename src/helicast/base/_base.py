from abc import ABC, abstractmethod
from functools import cached_property, partial
from typing import Any
from warnings import warn

import pandas as pd
import pydantic.dataclasses
from pydantic import validate_call
from sklearn.base import BaseEstimator as _SKLearnBaseEstimator
from typing_extensions import Self, dataclass_transform

from helicast._version import __version__
from helicast.typing import EstimatorMode
from helicast.utils import get_param_type_mapping, validate_equal_to_reference

__all__ = [
    "dataclass",
    "HelicastBaseEstimator",
    "validate_X",
    "validate_X_y",
    "validate_y",
]


def _series_to_dataframe(y: pd.Series) -> pd.DataFrame:
    """Convert a Series to a DataFrame. The Series must have a name that is a string."""
    if not isinstance(y.name, str):
        raise ValueError(
            f"pd.Series y should have a proper name (a string). "
            f"Found {y.name=} (type={type(y.name)})."
        )
    return y.to_frame(name=y.name)


def _validate_columns_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the labels of the columns are strings."""
    wrong_labels = [(c, type(c)) for c in df.columns if not isinstance(c, str)]
    if wrong_labels:
        raise ValueError(f"Columns labels must be strings. Found {wrong_labels=}.")
    return df


@validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
def validate_y(
    obj: object, y: pd.DataFrame | pd.Series | None, *, mode: EstimatorMode
) -> pd.DataFrame | None:
    if y is None:
        return None
    elif mode == EstimatorMode.FIT:
        if isinstance(y, pd.Series):
            y = _series_to_dataframe(y)
        obj.target_names_in_ = y.columns.tolist()
    elif mode == EstimatorMode.PREDICT:
        if isinstance(y, pd.Series):
            n_targets = len(obj.target_names_in_)
            if n_targets != 1:
                raise ValueError(f"{n_targets=} is not 1 but we got y as a pd.Series.")
            y = y.to_frame(name=obj.target_names_in_[0])
        columns = validate_equal_to_reference(
            y.columns, obj.target_names_in_, reorder=True
        )
        y = y[columns]
    else:
        raise RuntimeError(f"Validating y in {mode=}. This is unexpected!")
    return y


@validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
def validate_X(obj: object, X: pd.DataFrame, *, mode: EstimatorMode) -> pd.DataFrame:
    X = _validate_columns_labels(X)
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

    columns = X.columns
    # Set ``_disable_X_column_check = True`` in child class to disable this check
    if getattr(obj, "_disable_X_column_check", False) is False:
        columns = validate_equal_to_reference(X.columns, features, reorder=True)

    return X[columns]


def validate_X_y(
    obj: object,
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series | None = None,
    *,
    mode: EstimatorMode,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    X = validate_X(obj, X, mode=mode)
    y = validate_y(obj, y, mode=mode)

    if mode == EstimatorMode.FIT and y is not None:
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Found {len(X)=} and {len(y)=}."
            )
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same index!")

    return X, y


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

    if config is None:
        config = {"arbitrary_types_allowed": True}

    if config.get("validate_assignment", False) is True:
        raise RuntimeError(
            "Do not use 'validate_assignment=True' in the Pydantic config in child classes!"
        )

    return pydantic.dataclasses.dataclass(_cls=cls, config=config)


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

    ##################
    ### VALIDATION ###
    ##################

    def _validate_X(self, X: pd.DataFrame, *, mode: EstimatorMode) -> pd.DataFrame:
        return validate_X(self, X, mode=mode)

    def _validate_y(
        self, y: pd.DataFrame | pd.Series | None, *, mode: EstimatorMode
    ) -> pd.DataFrame | None:
        return validate_y(self, y, mode=mode)

    def _validate_X_y(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series | None = None,
        *,
        mode: EstimatorMode,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        return validate_X_y(self, X, y, mode=mode)

    ###################
    ### SKLEARN API ###
    ###################

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Get parameter names for the estimator. This replace the original sklearn
        implementation to ensure full compatibility with Pydantic dataclasses."""
        return list(get_param_type_mapping(cls).keys())

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
        X, y = self._validate_X_y(X, y, mode=EstimatorMode.FIT)
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
