import pandas as pd
import pytest
from typing_extensions import Self

from helicast.base import (
    HelicastBaseEstimator,
    InvertibleTransformerMixin,
    TransformerMixin,
    dataclass,
    is_fitted,
)


@dataclass
class DummyTransformer(TransformerMixin, HelicastBaseEstimator):
    a: int
    b: str
    c: float | None = None

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None, **kwargs) -> Self:
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()


@dataclass
class DummyInvertibleTransformer(InvertibleTransformerMixin, HelicastBaseEstimator):
    a: int
    b: str
    c: float | None = None

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None, **kwargs) -> Self:
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()

    def _inverse_transform(self, X):
        return X.copy()


@pytest.fixture
def transformer() -> DummyTransformer:
    return DummyTransformer(a=1, b="2")


@pytest.fixture
def invertible_transformer() -> DummyInvertibleTransformer:
    return DummyInvertibleTransformer(a=1, b="2")


@pytest.fixture
def all_transformers(transformer, invertible_transformer) -> list:
    return [transformer, invertible_transformer]


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_fit_transform(all_transformers, df):
    for tr in all_transformers:
        assert is_fitted(tr) is False
        tr.fit(df)
        df_tr = tr.transform(df)
        assert is_fitted(tr) is True
        assert isinstance(df_tr, pd.DataFrame)

        if isinstance(tr, DummyInvertibleTransformer):
            df_inv_tr = tr.inverse_transform(df_tr)
            assert isinstance(df_inv_tr, pd.DataFrame)


def test_fit_transform_with_adding_columns(all_transformers, df):
    # --- Check if adding a new column works (the new column should be ignored.)
    for tr in all_transformers:
        tr.fit(df)
        df_tr = tr.transform(df.assign(new_col=1.0))
        assert df_tr.columns.tolist() == df.columns.tolist()
        if isinstance(tr, DummyInvertibleTransformer):
            df_inv_tr = tr.inverse_transform(df_tr)
            assert df_inv_tr.columns.tolist() == df.columns.tolist()


def test_fit_transform_with_removing_columns(all_transformers, df):
    for tr in all_transformers:
        tr.fit(df)
    # --- Check if removing a column triggers an error
    new_df = df.drop(columns=df.columns[:1])
    for tr in all_transformers:
        with pytest.raises(ValueError):
            tr.transform(new_df)

        if isinstance(tr, DummyInvertibleTransformer):
            with pytest.raises(ValueError):
                tr.inverse_transform(new_df)
