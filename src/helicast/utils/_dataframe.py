from typing import Iterable, Union

import pandas as pd

from helicast.utils._collections import maybe_reorder_like

__all__ = [
    "maybe_reorder_columns",
]


def maybe_reorder_columns(
    X: pd.DataFrame, reference: Union[pd.DataFrame, Iterable], copy: bool = True
) -> pd.DataFrame:
    """Maybe reorder the columns of DataFrame ``X``. Reordering happens only if the
    columns of X are a subset of what is found in ``reference``.

    Args:
        X: Input DataFrame.
        reference: Reference DataFrame or iterable of column names.
        copy: If True, return a copy of the DataFrame. If False, return the DataFrame
            with the columns reordered inplace. Defaults to True.

    Returns:
        DataFrame with maybe reordered columns.
    """
    if isinstance(reference, pd.DataFrame):
        reference = reference.columns.to_list()

    columns = maybe_reorder_like(X.columns, reference)
    X = X[columns]
    if copy:
        return X.copy()
    return X
