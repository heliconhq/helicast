import sklearn.preprocessing

from helicast.scalers._base import BaseMixin, ColumnType


class MaxAbsScaler(BaseMixin, sklearn.preprocessing.MaxAbsScaler):
    _considered_types = ColumnType.NUMBER


class MinMaxScaler(BaseMixin, sklearn.preprocessing.MinMaxScaler):
    _considered_types = ColumnType.NUMBER


class QuantileTransformer(BaseMixin, sklearn.preprocessing.QuantileTransformer):
    _considered_types = ColumnType.NUMBER
