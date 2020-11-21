import logging

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from .utils import is_bool_series

logger = logging.getLogger(__name__)


class MixedDataStandardizer(TransformerMixin):

    def __init__(self, errors="ignore", inplace=False):
        self.sizes = {}
        self.types = None
        self.errors = errors
        self.inplace = inplace

    def _fit_categorical(self, series):
        n_classes = series.nunique()
        if n_classes < 2:
            msg = "Constant feature {} with value: {}".format(
                series.name, series.unique()
            )
            if self.errors == "ignore":
                logger.warning("Warning: " + msg)
            else:
                raise ValueError(msg)
        if series.isnull().sum():
            n_classes += 1
        self.types[series.name] = "category"
        self.sizes[series.name] = n_classes

    def fit(self, df):
        """Infer data types from a DataFrame."""
        self.types = df.dtypes
        is_numeric_type = df.dtypes.isin([np.dtype(float), np.dtype(int)])
        numerics = df.dtypes[is_numeric_type]
        objects = df.dtypes[df.dtypes == np.object_].index.tolist()
        for name, typ in numerics.iteritems():
            # Find booleans `hiding` as integers or floats
            if is_bool_series(df.loc[:, name]):
                self.types[name] = np.dtype(bool)
            # Find which integer dtypes are categorical
            # XXX: ATM integers are considered as label-encoded categoricals
            elif typ == np.dtype(int):
                self._fit_categorical(df.loc[:, name])
        for name in objects:
            self._fit_categorical(df.loc[:, name])
        return self

    def transform(self, df):
        if not self.inplace:
            df = df.copy()
        boolean_cols = self.types[self.types == "bool"].index
        df.loc[:, boolean_cols] = df.loc[:, boolean_cols].astype("bool")
        categorical_cols = self.types[self.types == "category"].index
        df.loc[:, categorical_cols] = df.loc[:, categorical_cols].apply(pd.Categorical)
        return df
