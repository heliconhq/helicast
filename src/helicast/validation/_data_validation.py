from typing import Annotated

import pandas as pd
from pydantic import TypeAdapter
from pydantic.functional_validators import AfterValidator

from helicast.typing._timeseries_dataframe import (
    CompleteTimeDataFrameValidator,
    TimeDataFrame,
)

__all__ = ["validate_timeseries_dataframe", "validate_complete_timeseries_dataframe"]


def validate_timeseries_dataframe(df: TimeDataFrame) -> pd.DataFrame:
    type_adapter = TypeAdapter(TimeDataFrame, config={"arbitrary_types_allowed": True})
    return type_adapter.validate_python(df)


def validate_complete_timeseries_dataframe(
    df: TimeDataFrame, frequency: pd.Timedelta | None = None
) -> pd.DataFrame:
    CustomTimeDataFrame = Annotated[
        pd.DataFrame,
        AfterValidator(CompleteTimeDataFrameValidator(allowed_frequency=frequency)),
    ]

    type_adapter = TypeAdapter(
        CustomTimeDataFrame, config={"arbitrary_types_allowed": True}
    )
    return type_adapter.validate_python(df)
