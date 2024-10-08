from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pydantic import validate_call
from sklearn.base import check_is_fitted
from sklearn.metrics import r2_score

from helicast.base._base import validate_X, validate_y
from helicast.typing import EstimatorMode

__all__ = [
    "PredictorMixin",
]


class PredictorMixin(ABC):
    """Mixin class for predictor (e.g., linear model) in Helicast.
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
        X = validate_X(self, X=X, mode=EstimatorMode.PREDICT)

        y_pred = self._predict(X)
        y_pred = validate_y(self, y=y_pred, mode=EstimatorMode.PREDICT, index=X.index)
        return y_pred

    def fit_predict(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None, **kwargs
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

    def score(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> float:
        y = validate_y(self, y=y, mode=EstimatorMode.PREDICT)
        y_pred = self.predict(X)
        score = r2_score(y, y_pred, sample_weight=sample_weight)
        return float(score)
