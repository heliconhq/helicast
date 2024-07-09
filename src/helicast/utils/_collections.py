from collections import Counter
from typing import Iterable, List

__all__ = [
    "maybe_reorder_like",
    "find_duplicates",
    "validate_no_duplicates",
    "validate_equal_to_reference",
    "validate_subset_of_reference",
    "check_no_missing_elements",
    "check_no_extra_elements",
]


def maybe_reorder_like(value: Iterable, reference: Iterable) -> List:
    """Maybe reorder the elements of ``value`` according to the ordering in
    ``reference``. Reordering happens only if the elements of ``value`` are a subset of
    ``reference``. Both ``value`` and ``reference`` must not contain duplicates.

    Args:
        value: Iterable to maybe reorder.
        reference: Reference iterable.

    Returns:
        Maybe reordered list.
    """
    validate_no_duplicates(value)
    validate_no_duplicates(reference)

    if set(value).issubset(set(reference)):
        return [c for c in reference if c in value]
    return list(value)


def find_duplicates(values: Iterable) -> List:
    """Find duplicates in an iterable ``values`` and return them as a list. If no
    duplicates are found, an empty list is returned."""
    duplicates = [item for item, count in Counter(values).items() if count > 1]
    return duplicates


def validate_no_duplicates(values: Iterable) -> Iterable:
    """Validate that an iterable ``values`` does not contain duplicates. If duplicates
    are found, a ``ValueError`` is raised."""
    duplicates = find_duplicates(values)
    if duplicates:
        raise ValueError(f"Iterable contains duplicates: {duplicates=}")
    return values


def validate_equal_to_reference(
    values: Iterable,
    reference: Iterable,
    reorder: bool = True,
    allow_duplicates: bool = False,
) -> List:
    """Validate that two iterables are equal to each other (order does not matter). The
    returned list will have the same order as `reference` if `reorder` is True.

    Args:
        values: Iterable to be validated.
        reference: Reference iterable to compare against.
        reorder: If True, the returned list will have the same order as `reference`.
            Defaults to True.
        allow_duplicates: If False, duplicates are not allowed in `values` and
            `reference`. Defaults to False.

    Raises:
        ValueError: If `values` is not equal to `reference`.

    Returns:
        List of values from `values` that are also in `reference`, following the order
        of `reference` if `reorder` is True.
    """
    # If they're equal, values is a subset with no missing elements!
    # This avoid duplicating some logic (but maybe slightly less optimal ¯\_(ツ)_/¯)
    results = validate_subset_of_reference(
        values=values,
        reference=reference,
        reorder=reorder,
        allow_duplicates=allow_duplicates,
    )
    check_no_missing_elements(results, reference)
    return results


def validate_subset_of_reference(
    values: Iterable,
    reference: Iterable,
    reorder: bool = True,
    allow_duplicates: bool = False,
) -> List:
    """Validate that `values` is a subset of `reference`. The returned list will have the
    same order as `reference` if `reorder` is True.

    Args:
        values: Iterable to be validated.
        reference: Reference iterable to compare against.
        reorder: If True, the returned list will have the same order as `reference`.
            Defaults to True.
        allow_duplicates: If False, duplicates are not allowed in `values` and
            `reference`. Defaults to False.

    Raises:
        ValueError: If `values` is not a subset of `reference`.

    Returns:
        List of values from `values` that are also in `reference`, following the order
        of `reference` if `reorder` is True.
    """
    if not allow_duplicates:
        validate_no_duplicates(values)
        validate_no_duplicates(reference)

    check_no_extra_elements(values, reference)

    if reorder:
        ordering = {item: i for i, item in enumerate(reference)}
        return sorted(values, key=lambda x: ordering[x])
    else:
        return list(values)


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
