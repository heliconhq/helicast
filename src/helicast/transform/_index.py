from copy import deepcopy
from typing import List, Union

import pandas as pd

from helicast.base import HelicastBaseEstimator, StatelessTransformerMixin, dataclass
from helicast.utils import link_docs_to_class

__all__ = [
    "IndexSorter",
    "sort_index",
    "IndexRenamer",
    "rename_index",
]


@dataclass
class IndexSorter(StatelessTransformerMixin, HelicastBaseEstimator):
    """Sort the index of a DataFrame.

    Args:
        ascending: If True, the index will be sorted in ascending order. If False, the
            index will be sorted in descending order. Defaults to True.
    """

    ascending: bool = True

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.ascending:
            is_sorted = X.index.is_monotonic_increasing
        else:
            is_sorted = X.index.is_monotonic_decreasing

        if not is_sorted:
            return X.sort_index(ascending=self.ascending)
        return X


@link_docs_to_class(cls=IndexSorter)
def sort_index(X: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    return IndexSorter(ascending=ascending).fit_transform(X)


@dataclass
class IndexRenamer(StatelessTransformerMixin, HelicastBaseEstimator):
    """Rename the index of a DataFrame.

    Args:
        name: Name to give to the index. If a list of names is provided, the index must
            have the same number of levels as the list. In this case, the index
            MultiIndex will be renamed with the provided names. If a single name is
            provided, the index will be renamed with this name.
    """

    name: Union[str, List[str]]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        names = deepcopy(self.name)
        if isinstance(names, str):
            names = [names]
        X = X.copy()
        X.index.names = names
        return X


@link_docs_to_class(cls=IndexRenamer)
def rename_index(X: pd.DataFrame, name: Union[str, List[str]]) -> pd.DataFrame:
    return IndexRenamer(name=name).fit_transform(X)
