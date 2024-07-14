from typing import Annotated

import pandas as pd
from pydantic.functional_validators import AfterValidator

from helicast.validation._pandas_timeseries_validation import DatetimeIndexValidator

__all__ = [
    "TimeDataFrame",
    "RegularTimeDataFrame",
    "TzAwareTimeDataFrame",
    "TzAwareRegularTimeDataFrame",
]


#: Annotated type for time-series DataFrame, that is a DataFrame with a DatetimeIndex.
#: The index can have any frequency and/or multiple frequencies.
TimeDataFrame = Annotated[pd.DataFrame, AfterValidator(DatetimeIndexValidator())]

#: Annotated type for time-series DataFrame with a unique frequency
RegularTimeDataFrame = Annotated[
    pd.DataFrame, AfterValidator(DatetimeIndexValidator(frequency=True))
]

#: Similar to TimeDataFrame, but timezones are enforced to be aware.
TzAwareTimeDataFrame = Annotated[
    pd.DataFrame, AfterValidator(DatetimeIndexValidator(tz="aware"))
]

#: Similar to RegularTimeDataFrame, but timezones are enforced to be aware.
TzAwareRegularTimeDataFrame = Annotated[
    pd.DataFrame, AfterValidator(DatetimeIndexValidator(frequency=True, tz="aware"))
]
