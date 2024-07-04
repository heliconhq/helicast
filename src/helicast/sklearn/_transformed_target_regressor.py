import pandas as pd
from sklearn.base import BaseEstimator as _SKLBaseEstimator
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor as _SKLTransformedTargetRegressor
from sklearn.pipeline import Pipeline as _SKLPipeline
from sklearn.utils import check_array
from sklearn.utils.metadata_routing import (
    _raise_for_unsupported_routing,
)

from helicast.sklearn._pipeline import Pipeline
from helicast.sklearn._wrapper import HelicastWrapper

__all__ = [
    "TransformedTargetRegressor",
]


class TransformedTargetRegressor(_SKLTransformedTargetRegressor):
    def __init__(self, regressor, transformer):
        if isinstance(regressor, _SKLPipeline):
            regressor = Pipeline(regressor.steps)
        elif isinstance(regressor, _SKLBaseEstimator):
            regressor = HelicastWrapper(regressor)
        else:
            raise TypeError(
                f"regressor must be a scikit-learn estimator, got {regressor}"
            )

        if isinstance(transformer, _SKLPipeline):
            transformer = Pipeline(transformer.steps)
        elif isinstance(transformer, _SKLBaseEstimator):
            transformer = HelicastWrapper(transformer)
        else:
            raise TypeError(
                f"transformer must be a scikit-learn estimator, got {transformer}"
            )

        super().__init__(regressor=regressor, transformer=transformer)

    def fit(self, X, y, **fit_params):
        ### HELICAST ###
        columns = y.columns.tolist()
        index = y.index
        ### HELICAST ###

        _raise_for_unsupported_routing(self, "fit", **fit_params)
        if y is None:
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                "requires y to be passed, but the target y is None."
            )
        y = check_array(
            y,
            input_name="y",
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype="numeric",
            allow_nd=True,
        )

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y

        ### HELICAST ###
        y_2d = pd.DataFrame(y_2d, columns=columns, index=index)
        ### HELICAST ###

        self._fit_transformer(y_2d)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.

        ### HELICAST: COMMENT ###
        # if y_trans.ndim == 2 and y_trans.shape[1] == 1:
        #    y_trans = y_trans.squeeze(axis=1)
        ### HELICAST: COMMENT ###

        if self.regressor is None:
            raise TypeError("The regressor has to be set to a regressor object.")
        else:
            self.regressor_ = clone(self.regressor)

        self.regressor_.fit(X, y_trans, **fit_params)

        if hasattr(self.regressor_, "feature_names_in_"):
            self.feature_names_in_ = self.regressor_.feature_names_in_

        return self
