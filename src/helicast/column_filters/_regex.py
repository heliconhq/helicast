import re
from logging import getLogger
from typing import List, Union

import numpy as np
import pandas as pd
from pydantic import field_validator

from helicast.column_filters._base import ColumnFilter
from helicast.logging import configure_logging

configure_logging()
logger = getLogger(__name__)

__all__ = [
    "RegexSelector",
    "select_columns_by_regex",
    "RegexRemover",
    "remove_columns_by_regex",
]


class RegexBase(ColumnFilter):
    """Base model for regex-based filtering."""

    patterns: Union[str, List[str]]

    @field_validator("patterns")
    @classmethod
    def auto_cast_to_list(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            v = [v]
        return v

    def _get_matched_columns(self, X: pd.DataFrame) -> List[str]:
        matched_columns = set()
        for pattern in self.patterns:
            local_match = [c for c in X.columns if re.search(pattern, c)]
            logger.info(f"For pattern '{pattern}', matched {local_match}.")
            matched_columns = matched_columns.union(set(local_match))

        matched_columns = list(matched_columns)
        logger.info(f"All unique columns matching: {matched_columns}.")
        return matched_columns

    def __or__(self, __value: object) -> ColumnFilter:
        if isinstance(__value, self.__class__):
            patterns = self.patterns + __value.patterns
            patterns = list(np.unique(patterns))
            return self.__class__(patterns=patterns)
        else:
            return ColumnFilter.__or__(self, __value)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, self.__class__):
            return set(self.patterns) == set(__value.patterns)
        return False


class RegexSelector(RegexBase):
    """ColumnFilter implementing a regex selection rule.

    Args:
        patterns: str of list of str that specifies the regex patterns. If a list of str
         is provided, a column will be matched if it matches with at least one of the
         patterns.
    """

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        return self._get_matched_columns(X)

    def __invert__(self):
        return RegexRemover(patterns=self.patterns)


def select_columns_by_regex(
    df: pd.DataFrame, patterns: Union[str, List[str]]
) -> pd.DataFrame:
    return RegexSelector(patterns=patterns).fit_transform(df)


class RegexRemover(RegexBase):
    """ColumnFilter implementing a regex exclusion rule.

    Args:
        patterns: str of list of str that specifies the regex patterns. If a list of str
         is provided, a column will be matched if it matches with at least one of the
         patterns.
    """

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        return list(set(X.columns) - set(self._get_matched_columns(X)))

    def __invert__(self):
        return RegexSelector(patterns=self.patterns)


def remove_columns_by_regex(
    df: pd.DataFrame, patterns: Union[str, List[str]]
) -> pd.DataFrame:
    return RegexRemover(patterns=patterns).fit_transform(df)
