from typing import Iterable

__all__ = [
    "check_no_missing_elements",
    "check_no_extra_elements",
]


def check_no_missing_elements(values: Iterable, reference: Iterable) -> None:
    """Ensure that there are no missing elements in ``values`` compared to ``reference``.
    If there are, raise a ValueError with the missing elements."""
    missing = set(reference) - set(values)
    if missing:
        raise ValueError(f"Missing elements: {missing}")


def check_no_extra_elements(values: Iterable, reference: Iterable) -> None:
    """Ensure that there are no extra elements in ``values`` compared to ``reference``.
    If there are, raise a ValueError with the extra elements."""
    extra = set(values) - set(reference)
    if extra:
        raise ValueError(f"Extra elements: {extra}")
