from abc import ABC, abstractmethod
from enum import auto
from functools import cached_property
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from pydantic import validate_call
from sklearn.base import BaseEstimator as _SKLearnBaseEstimator
from strenum import StrEnum
from typing_extensions import Self

from helicast.base import dataclass
from helicast.utils import get_param_type_mapping
from helicast.validation import validate_equal_to_reference

__all__ = [
    "BaseEstimator",
    "EstimatorMode",
]


class EstimatorMode(StrEnum):
    FIT = auto()
    TRANSFORM = auto()
    INVERSE_TRANSFORM = auto()
    PREDICT = auto()


@dataclass
class BaseEstimator(_SKLearnBaseEstimator, ABC):
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

    ##################
    ### VALIDATION ###
    ##################

    @validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
    def _validate_X(self, X: pd.DataFrame, *, mode: EstimatorMode) -> pd.DataFrame:
        # If fitting, store the feature names and return the input data
        if mode == EstimatorMode.FIT:
            self.feature_names_in_ = X.columns.tolist()
            return X

        # If transforming or predicting, validate the input data
        if mode == EstimatorMode.TRANSFORM or mode == EstimatorMode.PREDICT:
            features = self.feature_names_in_
        elif mode == EstimatorMode.INVERSE_TRANSFORM:
            features = self.feature_names_out_
        else:
            raise RuntimeError(f"Unexpected mode: {mode=}")

        columns = validate_equal_to_reference(X.columns, features, reorder=True)
        return X[columns]

    @validate_call(config={"arbitrary_types_allowed": True, "validate_return": True})
    def _validate_y(
        self, y: Union[pd.DataFrame, pd.Series, None], *, mode: EstimatorMode
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
            self.target_names_in_ = y.columns.tolist()
        elif mode == EstimatorMode.PREDICT:
            if isinstance(y, pd.Series):
                if len(self.target_names_in_) != 1:
                    raise ValueError(
                        f"{len(self.target_names_in_)=} is not 1 but we got y as a pd.Series."
                    )
                y = y.to_frame(name=self.target_names_in_[0])
            columns = validate_equal_to_reference(
                y.columns, self.target_names_in_, reorder=True
            )
            y = y[columns]
        else:
            raise RuntimeError(f"Validating y in {mode=}. This is unexpected!")
        return y

    def _validate_X_y(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, None] = None,
        *,
        mode: EstimatorMode,
    ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
        X = self._validate_X(X, mode=mode)
        y = self._validate_y(y, mode=mode)

        if mode == EstimatorMode.FIT and y is not None and len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Found {len(X)=} and {len(y)=}."
            )

        return X, y

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
        return self
