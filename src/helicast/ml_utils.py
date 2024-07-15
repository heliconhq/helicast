from itertools import pairwise
from typing import Tuple, Union

import numpy as np
import pandas as pd
from pydantic import PositiveFloat, validate_call

__all__ = [
    "split_data",
]


@validate_call(config={"arbitrary_types_allowed": True})
def split_data(
    obj: Union[pd.DataFrame, pd.Series],
    splits: Tuple[PositiveFloat],
    copy: bool = True,
) -> Union[Tuple[pd.DataFrame], Tuple[pd.Series]]:
    """Splits the data into the given splits, e.g., ``splits=(0.7, 0.15, 0.15)`` will
    split the data into 70%, 15%, and 15% (for, e.g., train, val, test).

    Args:
        obj: Pandas DataFrame or Series to split.
        splits: Split ratios. Should sum to 1, but it is normalized anyway. A custom
            number of splits can be used, e.g., ``(0.7, 0.10, 0.10, 0.10)``.
        copy: Whether to copy the data when splitting. Default is True.

    Returns:
        A tuple of the split data.
    """
    # Normalize and create the cumulative sum, e.g., [0.0, 0.7, 0.85, 1.0]
    splits = np.array(splits, dtype=float)
    splits = splits / np.sum(splits)
    splits = np.concatenate([[0.0], np.cumsum(splits)])
    # Transform to index
    splits = (splits * len(obj)).astype(int)
    # Constrain the two ends
    splits[0] = 0
    splits[-1] = len(obj)

    def _maybe_copy(obj):
        if copy:
            return obj.copy()
        return obj

    return tuple(_maybe_copy(obj.iloc[start:end]) for start, end in pairwise(splits))
