from typing import Any, List, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline as _SKLPipeline
from typing_extensions import Self

from helicast.sklearn._wrapper import helicast_auto_wrap
from helicast.utils import (
    check_method,
    is_fitable,
    is_invertible_transformer,
    is_predictor,
    is_transformer,
)

__all__ = [
    "Pipeline",
]


class Pipeline(_SKLPipeline):
    def __init__(self, steps: List[Tuple[str, Any]]):
        for i in range(len(steps)):
            steps[i] = (steps[i][0], helicast_auto_wrap(steps[i][1]))
        super().__init__(steps)

        try:
            self.set_output(transform="pandas")
        except Exception:
            pass

    ##################
    ### PROPERTIES ###
    ##################
    @property
    def _can_fit(self) -> bool:
        return all(is_fitable(step[1]) for step in self.steps)

    @property
    def _can_transform(self) -> bool:
        return all(is_transformer(step[1]) for step in self.steps)

    @property
    def _can_inverse_transform(self) -> bool:
        return all(is_invertible_transformer(step[1]) for step in self.steps)

    @property
    def _can_predict(self) -> bool:
        cond1 = all(is_transformer(step[1]) for step in self.steps[:-1])
        cond2 = is_predictor(self.steps[-1][1])
        return cond1 and cond2

    @property
    def feature_names_out_(self) -> List:
        if self._can_transform:
            return self.steps[-1][1].feature_names_out_
        else:
            raise AttributeError(
                "The pipeline is not a transformer, and therefore does not have "
                "the attribute 'feature_names_out_'"
            )

    @property
    def target_names_in_(self) -> List:
        if self._can_predict:
            return self.steps[0][1].target_names_in_
        else:
            raise AttributeError(
                "The pipeline is not a predictor, and therefore does not have "
                "the attribute 'target_names_in_'"
            )

    #######################
    ### SKLEARN METHODS ###
    #######################
    @check_method(check=is_fitable)
    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None, **fit_params
    ) -> Self:
        """Fit all the transformers one after the other and sequentially transform the
        data. Finally, fit the transformed data using the final estimator.

        Args:
            X: Training features pd.DataFrame.
            y: Training target pd.DataFrame or pd.Series. Can be None if the pipeline
                is not a predictor.
            **fit_params: Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.
        """
        return super().fit(X, y, **fit_params)

    @check_method(check=is_transformer)
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Call ``transform`` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        ``transform`` method. Only valid if the final estimator
        implements ``transform``.

        Args:
            X: Data to transform.

        Returns:
            Transformed data.
        """
        return super().transform(X)

    @check_method(check=is_invertible_transformer)
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply ``inverse_transform`` for each step in a reverse order. All estimators
        in the pipeline must support ``inverse_transform``.

        Args:
            X: Data to inverse transform.

        Returns:
            Inversed transformed data.
        """
        return super().inverse_transform(X)

    @check_method(check=is_predictor)
    def predict(self, X: pd.DataFrame, **predict_params) -> pd.DataFrame:
        """Transform the data, and apply ``predict`` with the final estimator. Call
        ``transform`` of each transformer in the pipeline. The transformed data are
        finally passed to the final estimator that calls ``predict`` method. Only valid
        if the final estimator implements ``predict``.

        Args:
            X: Training features pd.DataFrame

        Returns:
            Predicted target as pd.DataFrame.
        """
        return super().predict(X, **predict_params)

    def __sklearn_clone__(self):
        return Pipeline(
            steps=[(name, clone(estimator)) for name, estimator in self.steps]
        )
