from itertools import product
import copy

def find_and_change(config, param_name, param_value):
    for key, value in config.items():
        if type(value) is dict:
            find_and_change(value, param_name, param_value)
        else:
            if key == param_name:
                config[key] = param_value

    return config


def copy_and_change_config(config, varying_params):
    config = copy.deepcopy(config)
    for n, v in varying_params.items():
        find_and_change(config, n, v)

    return config


def get_all_v_params(varying_params, excluded_params={}):
    return [
        {
            k: p
            for k, p in zip(varying_params.keys(), params)
            if k not in excluded_params
        }
        for params in product(*varying_params.values())
    ]