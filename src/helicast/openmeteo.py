import json
from abc import ABC
from typing import Any, Dict, List, Literal, Tuple, Union

import pandas as pd
import requests
from joblib import expires_after
from pydantic import validate_call
from pydantic.dataclasses import dataclass
from requests import HTTPError
from requests.models import Response

from helicast.cache import cache
from helicast.typing import Timestamp
from helicast.utils import convert_float_seconds_to_timestamp, get_timezone

__all__ = [
    "OpenMeteoForecastAPI",
    "OpenMeteoHistoricalAPI",
    "OpenMeteoHistoricalForecastAPI",
    "get_openmeteo_forecast_data",
    "get_openmeteo_historical_data",
    "get_openmeteo_historical_forecast_data",
]

FREQUENCY_MAP = {
    "15min": "minutely_15",
    "1h": "hourly",
}


def _tz_localize_or_convert(value: pd.Timestamp, tz: str) -> pd.Timestamp:
    """Localize or convert a timestamp to the given timezone."""
    if value.tzinfo is None:
        return value.tz_localize(tz)
    else:
        return value.tz_convert(tz)


def _get_response(url: str, params: Union[Dict[str, Any], str]) -> Response:
    """Send a GET request to the given URL and return the response object.

    Args:
        url: The URL to send the request to.
        params: The parameters to send with the request. Can be a dictionary or a JSON
            string (which will be converted to a dictionary internally).

    Raises:
        HTTPError: If the status code is not 200.

    Returns:
        The response object.

    """
    if isinstance(params, str):
        params = json.loads(params)
    response = requests.get(url, params=params)
    if response.status_code != 200:
        reason = response.json().get("reason", "reason not found?!")
        raise HTTPError(
            f"Request failed with status code {response.status_code}. Reason: {reason}"
        )
    return response


def _parse_response(
    response: Response, frequency: str
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Returns a tuple with a dictionary of units and a DataFrame with the data."""
    data = response.json()

    # Clean the units
    units = data[f"{frequency}_units"]
    if units["time"] != "unixtime":
        raise ValueError(f"Expected time to be in unixtime, got {units['time']}")
    units = {k: f"[{v}]" for k, v in units.items()}

    df = pd.DataFrame(data[frequency])
    # Clean and sort the time index as a DateTimeIndex
    df.rename(columns={"time": "datetime"}, inplace=True)
    df["datetime"] = convert_float_seconds_to_timestamp(df["datetime"])
    df.set_index("datetime", drop=True, inplace=True)
    df.sort_index(inplace=True)

    # Add total radiation if needed
    radiations = {"diffuse_radiation_instant", "direct_radiation_instant"}
    if radiations.issubset(df.columns):
        df["total_radiation_instant"] = df[list(radiations)].sum(axis=1, skipna=False)
        units["total_radiation_instant"] = units["diffuse_radiation_instant"]

    return units, df


def _get_data(
    url: str, params: Union[Dict[str, Any], str]
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Returns a tuple with a dictionary of units and a DataFrame with the data."""
    if isinstance(params, dict):
        params = json.dumps(params)

    frequency = "hourly"
    if "minutely_15" in params:
        frequency = "minutely_15"

    response = _get_response(url, params)
    units, df = _parse_response(response, frequency)
    return units, df


#: Get the data cached indifinitely
_get_data_cached = cache(_get_data)

#: Get the data cached for 30 minutes
_get_data_cached_30min = cache(
    _get_data, cache_validation_callback=expires_after(minutes=30)
)


@dataclass
class BaseOpenMeteoAPI(ABC):
    latitude: float
    longitude: float
    tz: Union[str, None] = "UTC"

    def __post_init__(self):
        if self.tz is None:
            self.tz = get_timezone(self.latitude, self.longitude)

    @validate_call(config={"arbitrary_types_allowed": True})
    def _clean_dates(
        self, start_date: Timestamp, end_date: Union[Timestamp, None]
    ) -> Tuple[Timestamp, Timestamp]:
        """Clean the start and end dates, ensuring they are in the correct timezone and
        that the end date is not in the future (with a resolution of 1 hour)."""
        start_date = _tz_localize_or_convert(start_date, self.tz)

        if end_date is None:
            end_date = pd.Timestamp.now(tz=self.tz).floor("1h")
        end_date = _tz_localize_or_convert(end_date, self.tz)
        end_date = min(end_date, pd.Timestamp.now(tz=self.tz).floor("1h"))

        return start_date, end_date

    def _clean_params(self, **kwargs) -> Dict[str, Any]:
        """Clean the parameters for the API request. This includes converting the
        start_date and end_date to the correct format and timezone, and ensuring that
        the timeformat is "unixtime"."""
        if "start_date" in kwargs:
            start_date = kwargs.pop("start_date")
            end_date = kwargs.pop("end_date", None)

            start_date, end_date = self._clean_dates(start_date, end_date)

            kwargs["start_date"] = start_date.tz_convert("UTC").strftime("%Y-%m-%d")
            kwargs["end_date"] = end_date.tz_convert("UTC").strftime("%Y-%m-%d")

        if "timeformat" in kwargs and kwargs["timeformat"] != "unixtime":
            raise ValueError(
                f"Only 'unixtime' timeformat is supported. Found {kwargs['timeformat']}."
            )
        kwargs["timeformat"] = "unixtime"

        return kwargs


@dataclass
class OpenMeteoForecastAPI(BaseOpenMeteoAPI):
    def __post_init__(self):
        super().__post_init__()
        self.url = "https://api.open-meteo.com/v1/forecast"

    @validate_call(config={"arbitrary_types_allowed": True, "validate_default": True})
    def get_data(
        self,
        variables: List[str],
        frequency: Literal["15min", "1h"] = "1h",
        forecast_days: int = 7,
        past_days: int = 3,
        with_units: bool = False,
        truncate: bool = False,
    ) -> pd.DataFrame:
        # Convert the frequency to the format used by the API

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            FREQUENCY_MAP[frequency]: variables,
            "forecast_days": forecast_days,
            "past_days": past_days,
        }
        params = self._clean_params(**params)

        units, df = _get_data_cached_30min(self.url, params)

        if with_units:
            mapping = {k: f"{k}{units.get(k, '')}" for k in units.keys()}
            df.rename(columns=mapping, inplace=True)

        df.index = _tz_localize_or_convert(df.index, self.tz)
        df = df[sorted(df.columns)]
        df.dropna(axis=0, how="all", inplace=True)

        df = df[sorted(df.columns)]

        if truncate:
            df = df[df.index >= pd.Timestamp.now(tz=self.tz).floor(frequency)]

        return df


@dataclass
class OpenMeteoHistoricalAPI(BaseOpenMeteoAPI):
    def __post_init__(self):
        super().__post_init__()
        self.url = "https://archive-api.open-meteo.com/v1/archive"

    @validate_call(config={"arbitrary_types_allowed": True, "validate_default": True})
    def get_data(
        self,
        variables: List[str],
        start_date: Timestamp,
        frequency: Literal["1h"] = "1h",
        end_date: Union[Timestamp, None] = None,
        with_units: bool = False,
    ) -> pd.DataFrame:
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            FREQUENCY_MAP[frequency]: variables,
        }
        params = self._clean_params(**params)

        units, df = _get_data_cached(self.url, params)

        if with_units:
            mapping = {k: f"{k}{units.get(k, '')}" for k in units.keys()}
            df.rename(columns=mapping, inplace=True)

        df.index = _tz_localize_or_convert(df.index, self.tz)
        df = df[sorted(df.columns)]
        df.dropna(axis=0, how="all", inplace=True)

        df = df[sorted(df.columns)]

        start_date = _tz_localize_or_convert(start_date, self.tz)
        df = df[df.index >= start_date]

        return df


@dataclass
class OpenMeteoHistoricalForecastAPI(OpenMeteoHistoricalAPI):
    def __post_init__(self):
        super().__post_init__()
        self.url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    @validate_call(config={"arbitrary_types_allowed": True, "validate_default": True})
    def get_data(
        self,
        variables: List[str],
        start_date: Timestamp,
        frequency: Literal["1h", "15min"] = "1h",
        end_date: Union[Timestamp, None] = None,
        with_units: bool = False,
    ) -> pd.DataFrame:
        return super().get_data.raw_function(
            self,
            variables=variables,
            start_date=start_date,
            end_date=end_date,
            with_units=with_units,
            frequency=frequency,
        )


def get_openmeteo_forecast_data(
    latitude: float,
    longitude: float,
    variables: List[str],
    frequency: Literal["15min", "1h"] = "1h",
    forecast_days: int = 7,
    past_days: int = 3,
    with_units: bool = False,
    truncate: bool = False,
) -> pd.DataFrame:
    """Get the forecast data from the OpenMeteo API.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.
        variables: Variables to fetch. Refer to the OpenMeteo API documentation for the
            available variables.
        frequency: Frequency of the data to fetch. Can be "15min" or "1h". Defaults to
            "1h".
        forecast_days: Number of days to forecast. Defaults to 7.
        past_days: Number of past days to fetch. Defaults to 3.
        with_units: If True, appends the units to the column names. Defaults to False.
        truncate: If True, truncates the data to the current time. Defaults to False.

    Returns:
        The fetched data as a DataFrame with a UTC DateTimeIndex.
    """
    api = OpenMeteoForecastAPI(latitude=latitude, longitude=longitude)
    df = api.get_data(
        variables=variables,
        frequency=frequency,
        forecast_days=forecast_days,
        past_days=past_days,
        with_units=with_units,
        truncate=truncate,
    )
    return df


def get_openmeteo_historical_data(
    latitude: float,
    longitude: float,
    variables: List[str],
    start_date: Timestamp,
    frequency: Literal["15min", "1h"] = "1h",
    end_date: Union[Timestamp, None] = None,
    with_units: bool = False,
) -> pd.DataFrame:
    """Get the historical data from the OpenMeteo API.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.
        variables: Variables to fetch. Refer to the OpenMeteo API documentation for the
            available variables.
        start_date: Start date of the data to fetch.
        frequency: Frequency of the data to fetch. Can be "15min" or "1h". Defaults to
            "1h".
        end_date:  End date of the data to fetch. If None, fetches until the current
            time. Defaults to None.
        with_units: If True, appends the units to the column names. Defaults to False.

    Returns:
        The fetched data as a DataFrame with a UTC DateTimeIndex.
    """
    api = OpenMeteoHistoricalAPI(latitude=latitude, longitude=longitude)
    df = api.get_data(
        variables=variables,
        start_date=start_date,
        frequency=frequency,
        end_date=end_date,
        with_units=with_units,
    )
    return df


def get_openmeteo_historical_forecast_data(
    latitude: float,
    longitude: float,
    variables: List[str],
    start_date: Timestamp,
    frequency: Literal["15min", "1h"] = "1h",
    end_date: Union[Timestamp, None] = None,
    with_units: bool = False,
) -> pd.DataFrame:
    """_summary_

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.
        variables: Variables to fetch. Refer to the OpenMeteo API documentation for the
            available variables.
        start_date: Start date of the data to fetch.
        frequency: Frequency of the data to fetch. Can be "15min" or "1h". Defaults to
            "1h".
        end_date:  End date of the data to fetch. If None, fetches until the current
            time. Defaults to None.
        with_units: If True, appends the units to the column names. Defaults to False.

    Returns:
        The fetched data as a DataFrame with a UTC DateTimeIndex.
    """
    api = OpenMeteoHistoricalForecastAPI(latitude=latitude, longitude=longitude)
    df = api.get_data(
        variables=variables,
        start_date=start_date,
        frequency=frequency,
        end_date=end_date,
        with_units=with_units,
    )
    return df
