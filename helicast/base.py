import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, validate_call
from sklearn.base import BaseEstimator
from typing_extensions import Self

from helicast.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

_VALIDATE_CONFIG = {
    "arbitrary_types_allowed": True,
    "validate_return": True,
    "strict": True,
}

__all__ = [
    "PydanticBaseEstimator",
    "DFTransformer",
    "StatelessDFTransformer",
]


@validate_call(config=_VALIDATE_CONFIG)
def _validate_y(
    y: Union[pd.DataFrame, pd.Series, None] = None
) -> Union[pd.DataFrame, None]:
    """Validate that ``y`` is a pd.DataFrame, pd.Series or None. If ``y``, cast
    it onto a pd.DataFrame."""
    if isinstance(y, pd.Series):
        y = y.to_frame(name="values" if y.name is None else y.name)

    return y


@validate_call(config=_VALIDATE_CONFIG)
def _validate_X(X: pd.DataFrame) -> pd.DataFrame:
    """Validate that ``X`` is a pd.DataFrame."""
    return X


@validate_call(config=_VALIDATE_CONFIG)
def _validate_X_y(
    X: Union[pd.DataFrame],
    y: Union[pd.DataFrame, pd.Series, None] = None,
    same_length: bool = True,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    """Validate ``X`` and ``y``.

    Args:
        X: Feature matrix to validate. It must be a pd.DataFrame.
        y: Target matrix/vector to validate. It must be a pd.Series, pd.DataFrame or
            None (in which case no validation is performed). If ``y`` is a pd.Series,
            it will be cast onto a single column pd.DataFrame. Defaults to None.
        same_length: If ``True`` and ``y`` is not None, ``X`` and ``y`` are checked
            to have the same length (otherwise, raise a ``ValueError``). Defaults to
            True.

    Returns:
        The validated tuple ``X, y``.
    """
    X = _validate_X(X)
    y = _validate_y(y)

    if same_length and y is not None and len(y) != len(X):
        raise ValueError(
            f"`X` and `y` should have the same length. Found {len(X)=} and {len(y)=}"
        )

    return X, y


class PydanticBaseEstimator(BaseEstimator, BaseModel):
    """Extend the BaseEstimator from sklearn to include some Pydantic magic :)"""

    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, "model_config"):
            cls.model_config = ConfigDict()

        forced_defaults = {
            "arbitrary_types_allowed": True,
            "validate_assignment": True,
        }

        for k, v in forced_defaults.items():
            if k in cls.model_config.keys() and cls.model_config[k] != v:
                logger.warning(
                    (
                        f"{cls.__name__}: replacing {k}={cls.model_config[k]} in "
                        f"model_config to {k}={v}. This is necessary to ensure a proper "
                        f"behavior of PydanticBaseEstimator subclasses!"
                    )
                )
            cls.model_config[k] = v

        return super().__init_subclass__(**kwargs)

    @classmethod
    def _get_param_names(cls) -> List[str]:
        """Overriding sklearn BaseEstimator ``_get_param_names`` classmethod to
        accomodate for Pydantic BaseModel."""
        args = cls.model_fields
        args = {i: j for i, j in args.items() if j.init is not False}

        return list(sorted(args.keys()))

    def __str__(self) -> str:
        args = ", ".join(
            [f"{i}={str(getattr(self, i))}" for i in self.model_fields.keys()]
        )
        name = self.__class__.__name__
        return f"{name}({args})"


class DFTransformer(PydanticBaseEstimator, ABC):
    """Abstract base class for transformers that are stefull, i.e., that cannot call
    ``self.transform`` without a prior ``self.fit`` call.

    You need to implement the abstract methods

    .. code-block::python

        @abstractmethod
        def _fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, None], **kwargs) -> None:
            pass

        def _transform(
            self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
        ) -> pd.DataFrame:
            pass
    """

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "is_fitted_", False)

    @abstractmethod
    def _fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def _transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        pass

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> Self:
        X, y = _validate_X_y(X, y, same_length=True)
        self._fit(X, y, **kwargs)
        self.is_fitted_ = True
        return self

    def transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Not fitted!")

        X, y = _validate_X_y(X, y, same_length=True)
        X_new = self._transform(X, y, **kwargs)
        if not isinstance(X_new, pd.DataFrame):
            raise TypeError(f"{type(X_new)=} should be pd.DataFrame!")
        return X_new

    def fit_transform(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> pd.DataFrame:
        X, y = _validate_X_y(X, y, same_length=True)
        return self.fit(X, y, **kwargs).transform(X, y, **kwargs)


class StatelessDFTransformer(DFTransformer):
    """Abstract base class for transformers that are stateless, i.e., that can call
    ``self.transform`` without the need of a ``self.fit`` prior call.

    You need to implement the abstract method

    .. code-block::python

        def _transform(
            self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
        ) -> pd.DataFrame:
            pass
    """

    @property
    def is_fitted(self) -> bool:
        return True

    def _fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> None:
        pass

    def fit(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, None] = None, **kwargs
    ) -> Self:
        return self
