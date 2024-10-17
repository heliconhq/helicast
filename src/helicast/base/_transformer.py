from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from pydantic import validate_call
from sklearn.base import check_is_fitted

from helicast.base._base import validate_X
from helicast.typing import EstimatorMode

__all__ = [
    "TransformerMixin",
    "InvertibleTransformerMixin",
]


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
        X = validate_X(self, X, mode=EstimatorMode.TRANSFORM)
        X_tr = self._transform(X)
        self.feature_names_out_ = X_tr.columns.tolist()
        return X_tr

    def fit_transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, None] = None, **kwargs
    ) -> pd.DataFrame:
        """Syntatic sugar for calling ``fit`` and then ``transform``. The ``kwargs`` are
        passed to the ``fit`` method.

        Args:
            X: Input feature DataFrame.
            y: Input target DataFrame or Series, needed is the transformer is
                supervised. Defaults to None (unsupervised).

        Returns:
            The transformed DataFrame.
        """
        self.fit(X, y, **kwargs)
        return self.transform(X)


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
        X = validate_X(self, X, mode=EstimatorMode.INVERSE_TRANSFORM)
        X_inv_tr = self._inverse_transform(X)
        X_inv_tr = validate_X(self, X_inv_tr, mode=EstimatorMode.TRANSFORM)
        return X_inv_tr
