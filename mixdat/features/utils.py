import numpy as np


def is_bool_series(series):
    vals = np.sort(series.dropna().unique())
    if len(vals) != 2:
        return False
    else:
        # Values will be automatically downcast here
        bool_arr = np.array([0, 1], dtype=np.bool)
        return np.all(np.equal(vals, bool_arr))
