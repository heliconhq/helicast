from logging import getLogger
from typing import Union

import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor as _TransformedTargetRegressor
from sklearn.ensemble import AdaBoostRegressor as _AdaBoostRegressor
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.utils import check_array
from sklearn.utils.metadata_routing import _raise_for_unsupported_routing
from typing_extensions import Self

from helicast.logging import configure_logging
from helicast.models._sklearn import LinearRegression
from helicast.scalers._base import BaseMixin, ColumnType, _validate_X_y

configure_logging()
logger = getLogger(__name__)

__all__ = ["Pipeline", "TransformedTargetRegressor", "AdaBoostRegressor"]


class Pipeline(BaseMixin, _Pipeline):
    _considered_types = ColumnType.ALL


class AdaBoostRegressor(BaseMixin, _AdaBoostRegressor):
    _considered_types = ColumnType.ALL


class TransformedTargetRegressor(BaseMixin, _TransformedTargetRegressor):
    _considered_types = ColumnType.ALL

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series], **fit_params
    ) -> Self:
        """Fit the model according to the given training data.

        Args:
            X: Feature matrix as a pd.DataFrame of shape ``(n_samples, n_features)``,
             where ``n_samples`` is the number of samples and `n_features` is the number
             of features.
            y: Target matrix/vector as a pd.DataFrame or pd.Series of length
             ``n_samples``.

        Returns:
            Return the object itself, fitted.

        """

        ### CUSTOM, MODIFIED BY HELICAST: start
        X, y = _validate_X_y(X, y, same_length=True)
        self._register_names(X, y, if_name_exist="warn")
        columns = y.columns.to_list()
        index = y.index.copy()
        ### CUSTOM, MODIFIED BY HELICAST: end

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
        ### CUSTOM, MODIFIED BY HELICAST: start
        y_2d = pd.DataFrame(y_2d, columns=columns, index=index)
        ### CUSTOM, MODIFIED BY HELICAST: end
        self._fit_transformer(y_2d)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            ### CUSTOM, MODIFIED BY HELICAST: start
            pass
            # y_trans = y_trans.squeeze(axis=1)
            ### CUSTOM, MODIFIED BY HELICAST: start

        if self.regressor is None:
            ### CUSTOM, MODIFIED BY HELICAST: start
            # from sklearn.linear_model import LinearRegression
            # self.regressor_ = LinearRegression()
            self.regressor_ = LinearRegression()
            ### CUSTOM, MODIFIED BY HELICAST: end
        else:
            self.regressor_ = clone(self.regressor)

        self.regressor_.fit(X, y_trans, **fit_params)

        if hasattr(self.regressor_, "feature_names_in_"):
            self.feature_names_in_ = self.regressor_.feature_names_in_

        return self
