from typing import Any, List, Tuple

from sklearn.pipeline import Pipeline as _SKLPipeline

from helicast.base import (
    BaseEstimator,
)
from helicast.sklearn._wrapper import HelicastWrapper

__all__ = [
    "Pipeline",
]


class Pipeline(_SKLPipeline):
    def __init__(self, steps: List[Tuple[str, Any]]):
        for i in range(len(steps)):
            if not isinstance(steps[i][1], BaseEstimator):
                steps[i] = (steps[i][0], HelicastWrapper(steps[i][1]))
        super().__init__(steps)

    def feature_names_out_(self) -> List:
        if hasattr(self.steps[-1][1], "feature_names_out_"):
            return self.steps[-1][1].feature_names_out_
        elif hasattr(self.steps[-2][1], "feature_names_out_"):
            return self.steps[-2][1].feature_names_out_
        raise AttributeError(
            "The last step in the pipeline does not have the attribute 'feature_names_out_'"
        )

    def target_names_in_(self) -> List:
        if hasattr(self.steps[0][1], "target_names_in_"):
            return self.steps[0][1].target_names_in_
        raise AttributeError(
            "The first step in the pipeline does not have the attribute 'target_names_in_'"
        )

    def score(self, X, y, sample_weight=None):
        return self.steps[-1][1].score(X, y, sample_weight)
