from copy import deepcopy

import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor as _SKLTransformedTargetRegressor
from sklearn.pipeline import Pipeline as _SKLPipeline
from sklearn.utils import check_array
from sklearn.utils.metadata_routing import (
    _raise_for_unsupported_routing,
)

from helicast.base import validate_X_y
from helicast.sklearn._pipeline import Pipeline
from helicast.sklearn._wrapper import helicast_auto_wrap
from helicast.typing import EstimatorMode, InvertibleTransformerType, PredictorType

__all__ = [
    "TransformedTargetRegressor",
]


class TransformedTargetRegressor(_SKLTransformedTargetRegressor):
    def __init__(
        self, regressor: PredictorType, transformer: InvertibleTransformerType
    ):
        ### Regressor
        if isinstance(regressor, _SKLPipeline):
            regressor = Pipeline(regressor.steps)
            if not regressor._can_predict:
                raise TypeError(
                    f"Pipeline regressor is not a predictor ({regressor=})."
                )
        elif isinstance(regressor, PredictorType):
            regressor = helicast_auto_wrap(regressor)
        else:
            raise TypeError(f"regressor must be a predictor, got {regressor}")

        ### Transformer
        if isinstance(transformer, _SKLPipeline):
            transformer = Pipeline(transformer.steps)
            if not transformer._can_inverse_transform:
                raise TypeError(
                    f"Pipeline transformer is not an invertible transformer "
                    f"({transformer=})."
                )
        elif isinstance(transformer, InvertibleTransformerType):
            transformer = helicast_auto_wrap(transformer)
        else:
            raise TypeError(
                f"transformer must be an invertible transformer, got {transformer}"
            )

        super().__init__(regressor=regressor, transformer=transformer)

        try:
            self.set_output(transform="pandas")
        except Exception:
            pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series, **fit_params):
        ### HELICAST ###
        X, y = validate_X_y(self, X, y, mode=EstimatorMode.FIT)
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

        return self

    def __sklearn_clone__(self):
        return TransformedTargetRegressor(
            regressor=clone(self.regressor), transformer=clone(self.transformer)
        )

    def __deepcopy__(self, memo):
        new_instance = TransformedTargetRegressor(
            regressor=deepcopy(self.regressor, memo),
            transformer=deepcopy(self.transformer, memo),
        )
        for k in self.__dict__.keys():
            if k not in ["regressor", "transformer"]:
                new_instance.__dict__[k] = deepcopy(self.__dict__[k], memo)
        return new_instance
