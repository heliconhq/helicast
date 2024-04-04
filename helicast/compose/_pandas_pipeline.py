import logging
from typing import Union

import pandas as pd
from sklearn import config_context
from sklearn.pipeline import Pipeline
from typing_extensions import Self

from helicast.base import _validate_X_y
from helicast.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


__all__ = ["PandasPipeline"]


class PandasPipeline(Pipeline):
    """A sequence of data transformers with an optional final predictor.

    The ``PandasPipeline`` inherits from scikit-learn ``Pipeline`` and is quite similar
    except that it only accepts ``X`` as a pd.DataFrame and ``y`` as a pd.DataFrame or
    pd.Series (or ``None`` when ``y`` is not needed).
    For more information, check the scikit-learn documentation
    `sklearn.pipeline.Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_

    Args:
        steps: List of (name of step, estimator) tuples that are to be chained in
         sequential order. To be compatible with the scikit-learn API, all steps
         must define `fit`. All non-last steps must also define `transform`.
        memory: str or object with the joblib.Memory interface, default=None.
         Used to cache the fitted transformers of the pipeline. The last step
         will never be cached, even if it is a transformer. By default, no
         caching is performed. If a string is given, it is the path to the
         caching directory. Enabling caching triggers a clone of the transformers
         before fitting. Therefore, the transformer instance given to the
         pipeline cannot be inspected directly. Use the attribute ``named_steps``
         or ``steps`` to inspect estimators within the pipeline. Caching the
         transformers is advantageous when fitting is time consuming.
        verbose : bool, default=False.
         If True, the time elapsed while fitting each step will be printed as it
         is completed.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, None] = None,
        **fit_params,
    ) -> Self:
        X, y = _validate_X_y(X, y)

        if y is None:
            self.target_names_in_ = None
        else:
            self.target_names_in_ = y.columns.to_list()

        with config_context(transform_output="pandas"):
            super().fit(X, y, **fit_params)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X, _ = _validate_X_y(X, None)
        with config_context(transform_output="pandas"):
            return super().transform(X)

    def predict(self, X: pd.DataFrame, **predict_params) -> pd.DataFrame:
        X, _ = _validate_X_y(X, None)
        with config_context(transform_output="pandas"):
            y_hat = super().predict(X, **predict_params)

        if self.target_names_in_ is not None:
            columns = self.target_names_in_[:]
        else:
            if y_hat.shape[-1] == 1:
                columns = ["prediction"]
            else:
                columns = [f"prediction_{i}" for i in range(y_hat.shape[-1])]

        y_hat = pd.DataFrame(y_hat, columns=columns, index=X.index)
        return y_hat

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, None] = None,
        **fit_params,
    ) -> pd.DataFrame:
        with config_context(transform_output="pandas"):
            return self.fit(X, y, **fit_params).transform(X)
