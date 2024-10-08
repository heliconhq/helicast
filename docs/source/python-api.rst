.. _python-api:

====================
Python API
====================

.. toctree::
   :titlesonly:
   :caption: Base

   python-api/base/HelicastBaseEstimator.rst
   python-api/base/InvertibleTransformerMixin.rst
   python-api/base/PredictorMixin.rst
   python-api/base/StatelessEstimator.rst
   python-api/base/StatelessTransformerMixin.rst
   python-api/base/TransformerMixin.rst
   python-api/base/dataclass.rst
   python-api/base/is_stateless.rst
   python-api/base/validate_X.rst
   python-api/base/validate_X_y.rst
   python-api/base/validate_y.rst

.. toctree::
   :titlesonly:
   :caption: Cache

   python-api/cache/cache.rst
   python-api/cache/clear_cache.rst
   python-api/cache/CACHE_DIR.rst
   python-api/cache/MEMORY.rst

.. toctree::
   :titlesonly:
   :caption: Column Filters

   python-api/column_filters/AllSelector.rst
   python-api/column_filters/DTypeRemover.rst
   python-api/column_filters/DTypeSelector.rst
   python-api/column_filters/NameRemover.rst
   python-api/column_filters/NameSelector.rst
   python-api/column_filters/RegexRemover.rst
   python-api/column_filters/RegexSelector.rst
   python-api/column_filters/remove_columns_by_dtype.rst
   python-api/column_filters/remove_columns_by_names.rst
   python-api/column_filters/remove_columns_by_regex.rst
   python-api/column_filters/select_columns_by_dtype.rst
   python-api/column_filters/select_columns_by_names.rst
   python-api/column_filters/select_columns_by_regex.rst

.. toctree::
   :titlesonly:
   :caption: Datasets

   python-api/datasets/RegressionDataMaker.rst
   python-api/datasets/generate_dataframe.rst
   python-api/datasets/make_regression_data_by_r2_target.rst

.. toctree::
   :titlesonly:
   :caption: Logging

   python-api/logging/configure_logging.rst

.. toctree::
   :titlesonly:
   :caption: Meta Models

   python-api/meta_models/GeneralizedRegressor.rst

.. toctree::
   :titlesonly:
   :caption: Ml Utils

   python-api/ml_utils/split_data.rst

.. toctree::
   :titlesonly:
   :caption: Openmeteo

   python-api/openmeteo/OpenMeteoForecastAPI.rst
   python-api/openmeteo/OpenMeteoHistoricalAPI.rst
   python-api/openmeteo/OpenMeteoHistoricalForecastAPI.rst
   python-api/openmeteo/get_openmeteo_forecast_data.rst
   python-api/openmeteo/get_openmeteo_historical_data.rst
   python-api/openmeteo/get_openmeteo_historical_forecast_data.rst

.. toctree::
   :titlesonly:
   :caption: Sklearn

   python-api/sklearn/HelicastWrapper.rst
   python-api/sklearn/Pipeline.rst
   python-api/sklearn/TransformedTargetRegressor.rst
   python-api/sklearn/helicast_auto_wrap.rst

.. toctree::
   :titlesonly:
   :caption: Stateless Transform

   python-api/stateless_transform/Clipper.rst
   python-api/stateless_transform/SquareTransformer.rst

.. toctree::
   :titlesonly:
   :caption: Timeseries

   python-api/timeseries/DayOfWeekColumnAdder.rst
   python-api/timeseries/DayOfYearColumnAdder.rst
   python-api/timeseries/ForecastHorizonColumnsAdder.rst
   python-api/timeseries/HourColumnAdder.rst
   python-api/timeseries/SwedenPublicHolidayColumnAdder.rst
   python-api/timeseries/TzDatetimeIndexConverter.rst
   python-api/timeseries/TzDatetimeIndexLocalizator.rst
   python-api/timeseries/TzDatetimeIndexTransformer.rst
   python-api/timeseries/add_day_of_week_column.rst
   python-api/timeseries/add_day_of_year_column.rst
   python-api/timeseries/add_forecast_horizon_columns.rst
   python-api/timeseries/add_hour_column.rst
   python-api/timeseries/add_sweden_public_holiday_column.rst
   python-api/timeseries/apply_sun_mask.rst
   python-api/timeseries/get_sunrise_sunset_times.rst
   python-api/timeseries/tz_convert_datetime_index.rst
   python-api/timeseries/tz_localize_datetime_index.rst
   python-api/timeseries/tz_transform_datetime_index.rst

.. toctree::
   :titlesonly:
   :caption: Transform

   python-api/transform/FilteredTransformer.rst
   python-api/transform/FutureColumnsAdder.rst
   python-api/transform/IndexRenamer.rst
   python-api/transform/IndexSorter.rst
   python-api/transform/LaggedColumnsAdder.rst
   python-api/transform/add_future_columns.rst
   python-api/transform/add_lagged_columns.rst
   python-api/transform/rename_index.rst
   python-api/transform/sort_index.rst

.. toctree::
   :titlesonly:
   :caption: Typing

   python-api/typing/EstimatorMode.rst
   python-api/typing/InvertibleTransformerType.rst
   python-api/typing/PredictorType.rst
   python-api/typing/TransformerType.rst
   python-api/typing/UnsetType.rst
   python-api/typing/NegativeTimedelta.rst
   python-api/typing/NonNegativeTimedelta.rst
   python-api/typing/NonPositiveTimedelta.rst
   python-api/typing/PositiveTimedelta.rst
   python-api/typing/RegularTimeDataFrame.rst
   python-api/typing/TimeDataFrame.rst
   python-api/typing/Timedelta.rst
   python-api/typing/Timestamp.rst
   python-api/typing/TzAwareTimestamp.rst
   python-api/typing/TzNaiveTimestamp.rst
   python-api/typing/UNSET.rst

.. toctree::
   :titlesonly:
   :caption: Utils

   python-api/utils/TimeConverter.rst
   python-api/utils/adjust_series_forecast_timestamps.rst
   python-api/utils/are_timezones_equivalent.rst
   python-api/utils/auto_convert_to_datetime_index.rst
   python-api/utils/check_method.rst
   python-api/utils/check_no_extra_elements.rst
   python-api/utils/check_no_missing_elements.rst
   python-api/utils/convert_float_seconds_to_timestamp.rst
   python-api/utils/convert_timestamp_to_float_seconds.rst
   python-api/utils/find_common_indices.rst
   python-api/utils/find_datetime_index_unique_frequency.rst
   python-api/utils/find_duplicates.rst
   python-api/utils/get_classvar_list.rst
   python-api/utils/get_git_root.rst
   python-api/utils/get_param_type_mapping.rst
   python-api/utils/get_timezone.rst
   python-api/utils/has_method.rst
   python-api/utils/is_classvar.rst
   python-api/utils/is_fitable.rst
   python-api/utils/is_invertible_transformer.rst
   python-api/utils/is_predictor.rst
   python-api/utils/is_transformer.rst
   python-api/utils/iterate_days.rst
   python-api/utils/link_docs_to_class.rst
   python-api/utils/maybe_reorder_columns.rst
   python-api/utils/maybe_reorder_like.rst
   python-api/utils/numpy_array_to_dataframe.rst
   python-api/utils/restrict_to_common_indices.rst
   python-api/utils/select_day.rst
   python-api/utils/series_to_dataframe.rst
   python-api/utils/set_method_flags.rst
   python-api/utils/validate_column_names_as_string.rst
   python-api/utils/validate_equal_to_reference.rst
   python-api/utils/validate_no_duplicates.rst
   python-api/utils/validate_subset_of_reference.rst

.. toctree::
   :titlesonly:
   :caption: Validation

   python-api/validation/ColumnNamesValidator.rst
   python-api/validation/DatetimeIndexValidator.rst
   python-api/validation/ScalarToListValidator.rst
   python-api/validation/TzAwareDatetimeIndexValidator.rst
   python-api/validation/TzNaiveDatetimeIndexValidator.rst
   python-api/validation/validate_datetime_index.rst
   python-api/validation/validate_scalar_to_list.rst
   python-api/validation/validate_tz_aware_datetime_index.rst
   python-api/validation/validate_tz_naive_datetime_index.rst


