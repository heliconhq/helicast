import random
import string
from datetime import datetime, timedelta
from random import sample
from typing import Literal

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass

__all__ = [
    "generate_dataframe",
]


def generate_random_string(
    min_length: int,
    max_length: int,
    digits: bool = False,
    punctuation: bool = False,
) -> str:
    characters = string.ascii_letters
    if digits:
        characters += string.digits
    if punctuation:
        characters += string.punctuation

    length = random.randint(min_length, max_length)
    return "".join(random.choices(characters, k=length))


@dataclass
class FloatSeriesGenerator:
    precision: Literal[64, 32] = 64

    def __call__(self, n: int) -> pd.Series:
        data = np.random.normal(size=(n,)).astype(np.float64)
        return pd.Series(data).astype(f"float{self.precision}")


@dataclass
class IntSeriesGenerator:
    precision: Literal[64, 32] = 64

    def __call__(self, n: int) -> pd.Series:
        data = np.random.randint(-1000, 1000, size=(n,)).astype(np.int64)
        return pd.Series(data).astype(f"int{self.precision}")


@dataclass
class UIntSeriesGenerator:
    precision: Literal[64, 32] = 64

    def __call__(self, n: int) -> pd.Series:
        data = np.random.randint(0, 1000, size=(n,)).astype(np.uint64)
        return pd.Series(data).astype(f"uint{self.precision}")


@dataclass
class DatetimeSeriesGenerator:
    start: datetime = datetime(2023, 1, 1)
    frequency: timedelta = timedelta(days=1)
    tz: str | None = None

    def __call__(self, n: int) -> pd.Series:
        data = pd.date_range(start=self.start, periods=n, freq=self.frequency)
        if self.tz is not None:
            data = data.tz_localize("UTC").tz_convert(self.tz)
        return data


@dataclass
class IrregularDatetimeSeriesGenerator:
    tz: str | None = None

    def __call__(self, n: int) -> pd.Series:
        if n < 3:
            raise ValueError("n must be at least 3 for irregular datetime series!")
        years = np.random.randint(2020, 2023, size=(n,))
        month = np.random.randint(1, 12, size=(n,))
        days = np.random.randint(1, 28, size=(n,))
        hours = np.random.randint(0, 23, size=(n,))
        minutes = np.random.randint(0, 59, size=(n,))
        seconds = np.random.randint(0, 59, size=(n,))
        data = [
            datetime(*args) for args in zip(years, month, days, hours, minutes, seconds)
        ]
        data = pd.Series(data).sort_values(ignore_index=True)
        if self.tz is not None:
            data = data.dt.tz_localize("UTC").dt.tz_convert(self.tz)
        return data


@dataclass
class StringSeriesGenerator:
    min_length: int = 0
    max_length: int = 10
    digits: bool = False
    punctuation: bool = False

    def __call__(self, n: int) -> pd.Series:
        data = [
            generate_random_string(
                self.min_length,
                self.max_length,
                digits=self.digits,
                punctuation=self.punctuation,
            )
            for _ in range(n)
        ]

        for i in [self.min_length, self.max_length]:
            index = random.randint(0, n - 1)
            data[index] = generate_random_string(i, i)
        return pd.Series(data).astype(np.str_)


_FLOATS = {f"float{i}": FloatSeriesGenerator(i) for i in [64, 32]}
_INTS = {f"int{i}": IntSeriesGenerator(i) for i in [64, 32]}
_UINTS = {f"uint{i}": UIntSeriesGenerator(i) for i in [64, 32]}

_TYPES = dict(
    **_FLOATS,
    **_INTS,
    **_UINTS,
    **{
        "datetime": DatetimeSeriesGenerator(),
        "datetimetz": DatetimeSeriesGenerator(tz="Europe/Stockholm"),
        "datetimeutc": DatetimeSeriesGenerator(tz="UTC"),
        "irregular-datetime": IrregularDatetimeSeriesGenerator(),
        "irregular-datetimetz": IrregularDatetimeSeriesGenerator(tz="Europe/Stockholm"),
        "irregular-datetimeutc": IrregularDatetimeSeriesGenerator(tz="UTC"),
        "string": StringSeriesGenerator(),
        "stringdigit": StringSeriesGenerator(digits=True),
        "stringpunct": StringSeriesGenerator(punctuation=True),
    },
)


def generate_dataframe(
    n_rows: int = 100,
    n_cols: int = 20,
    dtypes: list[str] | None = None,
    index_freq: str | None = None,
    pyarrow_backend: bool = False,
) -> pd.DataFrame:
    """Generate a random pd.DataFrame with the specified number of rows and (maybe)
    columns. The dtypes can be ["number", "float", "int", "uint", "datetime",
    "irregular_datetime", "string"].

    Args:
        n_row: Number of rows
        n_cols: Number of columns. If the types provided in ``dtypes`` are such that
            ``n_cols`` is less than the number of unique types, the function will
            the returned DataFrame will have more columns than ``n_cols``.
        dtypes: List of types to generate. The types can be ["number", "float", "int",
            "uint", "datetime", "irregular_datetime", "string"]. If None, all types
            will be generated.
        index_freq: Frequency of the index if a datetime index is required, e.g.,
            "1D", "30min", ... If None, the DataFrame will have a RangeIndex. Defaults
            to None
        pyarrow_backend: If True, the returned DataFrame will have a pyarrow backend.
            Defaults to False.

    Returns:
        A randomly generated pd.DataFrame.
    """
    if dtypes is None:
        dtypes = list(_TYPES.keys())

    dtypes = list(set(dtypes))
    if "number" in dtypes:
        dtypes.remove("number")
        dtypes.extend(["float", "int", "uint"])

    if "float" in dtypes:
        dtypes.remove("float")
        dtypes.extend(list(_FLOATS.keys()))
    if "int" in dtypes:
        dtypes.remove("int")
        dtypes.extend(list(_INTS.keys()))
    if "uint" in dtypes:
        dtypes.remove("uint")
        dtypes.extend(list(_UINTS.keys()))

    invalid_types = [i for i in dtypes if i not in _TYPES.keys()]
    if invalid_types:
        raise ValueError(f"Invalid types: {invalid_types}")

    if n_cols > len(dtypes):
        choices = np.random.choice(dtypes, size=(n_cols - len(dtypes)), replace=True)
        dtypes.extend(list(choices))

    # Shuffle randomly the dtypes
    dtypes = sample(dtypes, k=len(dtypes))

    data = pd.DataFrame(
        {f"column_{dtype}_{i}": _TYPES[dtype](n_rows) for i, dtype in enumerate(dtypes)}
    )

    # Set datetime index if `index_freq` is required
    if index_freq is not None:
        index = pd.date_range(start="2023-01-01", periods=n_rows, freq=index_freq)
        data = data.set_index(index)
        data.index.name = "datetime"

    if pyarrow_backend:
        data = data.convert_dtypes(dtype_backend="pyarrow")
    return data
