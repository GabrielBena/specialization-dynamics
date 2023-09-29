import os
import matplotlib.pyplot as plt
import numpy as np


def set_style():
    file_path = os.path.realpath(__file__)
    file_path = file_path.replace("plotting.py", "style_sheet.mplstyle")
    plt.style.use(file_path)


def single_filter(data, key, value):
    if key[0] == "!":
        if value is None:
            return ~data[key[1:]].isnull()
        else:
            return data[key[1:]] != value
    else:
        if value is None:
            return data[key].isnull()
        else:
            return data[key] == value


def filter_data(data, v_params):
    data = data.copy()
    filters = []
    for key, value in v_params.items():
        if key in data.columns or (key[0] == "!" and key[1:] in data.columns):
            if isinstance(value, list):
                filter = np.sum(
                    [single_filter(data, key, v) for v in value], axis=0
                ).astype(bool)
            else:
                filter = single_filter(data, key, value)

        filters.append(filter)

    filter = np.prod(filters, axis=0).astype(bool)
    data = data[filter]
    return data, filter
