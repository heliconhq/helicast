from collections import Counter
from typing import Iterable, List

__all__ = [
    "find_duplicates",
]


def find_duplicates(values: Iterable) -> List:
    """Find duplicates in an iterable ``values`` and return them as a list. If no
    duplicates are found, an empty list is returned."""
    duplicates = [item for item, count in Counter(values).items() if count > 1]
    return duplicates
