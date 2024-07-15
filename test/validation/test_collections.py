import pytest
from helicast.utils import (
    validate_equal_to_reference,
    validate_no_duplicates,
    validate_subset_of_reference,
)


# Parametrized tests for validate_no_duplicates that succeeds
@pytest.mark.parametrize("values", [[1, 2, 3, 4, 5], (1, 2, 3, 4, 5)])
def test_validate_no_duplicates(values):
    validate_no_duplicates(values)


# Parametrized tests for validate_no_duplicates that fails
@pytest.mark.parametrize("values", [[1, 2, 2, 4, 5], (1, 2, 2, 4, 5)])
def test_validate_no_duplicates_raises_error(values):
    with pytest.raises(ValueError):
        validate_no_duplicates(values)


# Parametrized tests for validate_equal_to_reference
@pytest.mark.parametrize(
    "values, reference, reorder, allow_duplicates, expected",
    [
        ([1, 2, 3], [3, 1, 2], True, False, [3, 1, 2]),
        ([1, 2, 3], [3, 1, 2], True, True, [3, 1, 2]),
        ([1, 2, 3], [3, 1, 2], False, True, [1, 2, 3]),
        ([1, 2, 1, 3, 3], [3, 1, 2], True, True, [3, 3, 1, 1, 2]),
        ([1, 2, 1, 3, 3], [3, 1, 2], False, True, [1, 2, 1, 3, 3]),
    ],
)
def test_validate_equal_to_reference(
    values, reference, reorder, allow_duplicates, expected
):
    value = validate_equal_to_reference(values, reference, reorder, allow_duplicates)
    assert value == expected


@pytest.mark.parametrize(
    "values, reference, allow_duplicates, exception",
    [
        ([1, 2], [1, 2, 3], False, ValueError),
        ([1, 2, 3, 4], [1, 2, 3], False, ValueError),
        ([1, 2, 2, 3], [1, 2, 3], False, ValueError),
    ],
)
def test_validate_equal_to_reference_raises_error(
    values, reference, allow_duplicates, exception
):
    with pytest.raises(exception):
        validate_equal_to_reference(
            values, reference, allow_duplicates=allow_duplicates
        )


# Parametrized tests for validate_subset_of_reference
@pytest.mark.parametrize(
    "values, reference, reorder, expected",
    [
        ([2, 3], [3, 2, 1], True, [3, 2]),
        ([2, 3], [3, 2, 1], False, [2, 3]),
    ],
)
def test_validate_subset_of_reference(values, reference, reorder, expected):
    assert validate_subset_of_reference(values, reference, reorder) == expected


@pytest.mark.parametrize(
    "values, reference, exception",
    [
        ([1, 4], [1, 2, 3], ValueError),
        ([1, 2, 3, 4], [1, 2, 3], ValueError),
    ],
)
def test_validate_subset_of_reference_raises_error(values, reference, exception):
    with pytest.raises(exception):
        validate_subset_of_reference(values, reference)
