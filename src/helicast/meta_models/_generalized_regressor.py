import pandas as pd
from pydantic import validate_call

from helicast.sklearn import helicast_auto_wrap
from helicast.sklearn._transformed_target_regressor import TransformedTargetRegressor
from helicast.typing import InvertibleTransformerType, PredictorType, TransformerType

__all__ = [
    "GeneralizedRegressor",
]


class GeneralizedRegressor(TransformedTargetRegressor):
    """Generalized regressor that extends the ``TransformedTargetRegressor``. It adds
    a ``predict_transformer`` that can apply transformations to the target variables
    during ``self.predict(...)``.

    As for the ``TransformedTargetRegressor``, the target variables are transformed
    using ``target_transformer.fit_transform(...)`` before passing it to the regressor.
    During ``self.predict(...)``:
    1. The regressor predicts the scaled transformed targets;
    2. The scaled transformed targets are transformed back using
    ``target_transformer.inverse_transform``
    3. The targets are transformed using the ``predict_transformer.transform(...)``

    Args:
        regressor: Regressor to be used.
        transformer: Transformer to be used on the target variables similar to
            ``TransformedTargetRegressor``.
        predict_transformer: Transformer to be used after predicting, typically to
            apply some constraints (e.g., clip the values). This transformer should
            be stateless. If None, no transformation is applied. Defaults to None.
    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        regressor: PredictorType,
        transformer: InvertibleTransformerType,
        predict_transformer: TransformerType | None = None,
    ):
        super().__init__(regressor, transformer)

        if predict_transformer is not None:
            predict_transformer = helicast_auto_wrap(predict_transformer)
        self.predict_transformer = predict_transformer

    def predict(self, X: pd.DataFrame, **predict_params) -> pd.DataFrame:
        y_pred = super().predict(X, **predict_params)
        if self.predict_transformer is not None:
            y_pred = self.predict_transformer.transform(y_pred)
        return y_pred
