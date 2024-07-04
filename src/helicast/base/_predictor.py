from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from pydantic import validate_call
from sklearn.base import check_is_fitted
from sklearn.metrics import r2_score

from helicast.base import EstimatorMode

__all__ = ["PredictorMixin"]


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
