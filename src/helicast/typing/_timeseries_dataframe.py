from typing import Annotated, Callable, Union

import pandas as pd
from pydantic.functional_validators import AfterValidator

__all__ = [
    "TimeDataFrame",
    "RegularTimeDataFrame",
]


def _validate_time_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    return df


def _validate_regular_time_dataframe(
    frequency: Union[bool, pd.Timedelta],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        df = _validate_time_dataframe(df)

        if frequency is False:
            return df

        diff = (df.index[1:] - df.index[:-1]).unique()

        if frequency is True:
            if len(diff) > 1:
                raise ValueError(
                    f"DataFrame must have a unique frequency. "
                    f"Found unique frequencies: {diff}."
                )
        elif isinstance(frequency, pd.Timedelta):
            if len(diff) > 1:
                raise ValueError(
                    f"DataFrame must have a frequency of {frequency}. "
                    f"Found unique frequencies: {diff}."
                )
            elif diff[0] != frequency:
                raise ValueError(
                    f"DataFrame must have a frequency of {frequency}. "
                    f"Found {diff[0]} instead."
                )
        else:
            raise ValueError(f"Invalid frequency: {frequency}")
        return df

    return _validate


#: Annotated type for time-series DataFrame, that is a DataFrame with a DatetimeIndex.
#: The index can have any frequency and/or multiple frequencies.
TimeDataFrame = Annotated[pd.DataFrame, AfterValidator(_validate_time_dataframe)]

#: Annotated type for time-series DataFrame with a unique frequency
RegularTimeDataFrame = Annotated[
    pd.DataFrame, AfterValidator(_validate_regular_time_dataframe(True))
]
