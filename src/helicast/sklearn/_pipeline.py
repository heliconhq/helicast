from typing import Any, List, Tuple

from sklearn.pipeline import Pipeline as _SKLPipeline

from helicast.sklearn._wrapper import helicast_auto_wrap

__all__ = [
    "Pipeline",
]


class Pipeline(_SKLPipeline):
    def __init__(self, steps: List[Tuple[str, Any]]):
        for i in range(len(steps)):
            steps[i] = (steps[i][0], helicast_auto_wrap(steps[i][1]))
        super().__init__(steps)

    @property
    def feature_names_out_(self) -> List:
        if hasattr(self.steps[-1][1], "feature_names_out_"):
            return self.steps[-1][1].feature_names_out_
        elif hasattr(self.steps[-2][1], "feature_names_out_"):
            return self.steps[-2][1].feature_names_out_
        raise AttributeError(
            "The last step in the pipeline does not have the attribute 'feature_names_out_'"
        )

    @property
    def target_names_in_(self) -> List:
        if hasattr(self.steps[0][1], "target_names_in_"):
            return self.steps[0][1].target_names_in_
        raise AttributeError(
            "The first step in the pipeline does not have the attribute 'target_names_in_'"
        )
