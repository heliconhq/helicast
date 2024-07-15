from typing import Union

import pandas as pd
import requests
from pydantic import validate_call

from helicast.cache import cache

__all__ = [
    "get_sunrise_sunset_times",
    "apply_sun_mask",
]


@cache
def _request_sunrise_sunset_times(
    latitude: str, longitude: str, start_date: str, end_date: str
) -> pd.DataFrame:
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["sunrise", "sunset"],
        "timezone": "GMT",
    }

    df = pd.DataFrame(requests.get(url, params=params).json()["daily"])
    for c in df.columns:
        df[c] = pd.to_datetime(df[c]).dt.tz_localize("GMT")
    return df


@validate_call(config={"arbitrary_types_allowed": True})
def get_sunrise_sunset_times(
    latitude: Union[str, float],
    longitude: Union[str, float],
    start_date: Union[pd.Timestamp, str],
    end_date: Union[pd.Timestamp, str],
) -> pd.Series:
    """Returns a series with value ``1`` for sunrise times and ``0`` for sunset times.
    Can be used later to mask the data with the sun's presence (e.g., we don't expect
    solar power generation during the night).

    .. code-block:: python

        2024-03-01 05:45:00+00:00    1
        2024-03-01 16:27:00+00:00    0
        2024-03-02 05:42:00+00:00    1
        ...
        2024-07-10 02:18:00+00:00    1
        2024-07-10 19:41:00+00:00    0
        Name: is_sun_out, Length: 264, dtype: int64

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location
        start_date: Start date of the forecast in the format ``YYYY-MM-DD``.
        end_date: End date of the forecast in the format ``YYYY-MM-DD``.

    Returns:
        A series with value ``1`` for sunrise times and ``0`` for sunset times.
    """
    latitude = f"{latitude:.6f}" if isinstance(latitude, float) else latitude
    longitude = f"{longitude:.6f}" if isinstance(longitude, float) else longitude
    start_date = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(end_date).strftime("%Y-%m-%d")

    df = _request_sunrise_sunset_times(latitude, longitude, start_date, end_date)
    sunrise = pd.Series(1, index=df["sunrise"], name="is_sun_out")
    sunset = pd.Series(0, index=df["sunset"], name="is_sun_out")
    return pd.concat([sunrise, sunset]).sort_index()


@validate_call(config={"arbitrary_types_allowed": True})
def apply_sun_mask(
    data: Union[pd.Series, pd.DataFrame], sunset_sunrise_times: pd.Series
) -> Union[pd.Series, pd.DataFrame]:
    """Using the sunrise and sunset times (can be obtained with
    :func:`get_sunrise_sunset_times`), masks the data with the sun's presenc (1 for day,
    0 for night).

    Args:
        data: Data to mask.
        sunset_sunrise_times: Sunrise and sunset times.

    Returns:
        Masked data
    """
    # Check that the date range is covered!
    if sunset_sunrise_times.index.min() > data.index.min():
        raise ValueError("sunset_sunrise_times starts after the data.")
    if sunset_sunrise_times.index.max() < data.index.max():
        raise ValueError("sunset_sunrise_times ends before the data.")

    sun_mask = sunset_sunrise_times.reindex(data.index, method="ffill")
    if isinstance(data, pd.Series):
        return data * sun_mask
    elif isinstance(data, pd.DataFrame):
        return data.mul(sun_mask, axis=0)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
