import torch
import pandas as pd
from tqdm.notebook import tqdm as tqdm_n
from tqdm.notebook import tqdm
from itertools import product
import copy
from hashlib import sha1
from dynspec.models import init_model
from dynspec.training import train_community
from dynspec.metrics import compute_retraining_metric, global_diff_metric


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


def is_notebook():
    try:
        get_ipython()
        notebook = True
    except NameError:
        notebook = False
    return notebook


class Experiment(object):
    def __init__(
        self,
        default_config,
        varying_params,
        load_save,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_tests=1,
    ) -> None:
        self.varying_params = varying_params
        self.all_varying_params = get_all_v_params(varying_params)
        self.default_config = default_config
        self._models, self.optimizers = [], []

        if load_save:
            self.results, self.result_path = self.load_result_df()
        else:
            print("Creating new Experiment")
            self.loaded = False
            self.results = {}

        if isinstance(self.results, dict):
            for v_p in self.all_varying_params:
                for _ in range(n_tests):
                    for n, v in v_p.items():
                        self.results.setdefault(n, []).append(v)
                    self.results.setdefault("varying_params", []).append(v_p)

                    config = copy_and_change_config(default_config, v_p)

                    model, optimizer = init_model(config, device)
                    self._models.append(model)
                    self.optimizers.append(optimizer)
            self.results = pd.DataFrame.from_dict(self.results)

        else:
            for state_dict, config in zip(
                self.results["state_dicts"].values, self.all_configs
            ):
                model, optimizer = init_model(config, device)
                model.load_state_dict(state_dict)
                self._models.append(model)
                self.optimizers.append(optimizer)

    def load_result_df(self):
        v_hash = sha1(str(self.varying_params).encode("utf-8")).hexdigest()
        load_path = f"../results/example/{v_hash}"
        try:
            results = pd.read_pickle(load_path)
            self.loaded = True
            print("Results loaded")
        except FileNotFoundError:
            results = {}
            self.loaded = False
            print("No results found, creating new dict")

        return results, load_path

    def save_result_df(self):
        if not hasattr(self, "result_path"):
            v_hash = sha1(str(self.varying_params).encode("utf-8")).hexdigest()
            self.result_path = f"../results/example/{v_hash}"

        self.results.to_pickle(self.result_path)
        print("Results saved")

    def run(
        self,
        loaders,
        save=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        train_results = []
        tqdm_f = tqdm_n if is_notebook() else tqdm
        exp_bar = tqdm_f(self.models)
        stop_acc = kwargs.pop("stop_acc", self.default_config["training"]["stop_acc"])
        n_epochs = kwargs.pop("n_epochs", self.default_config["training"]["n_epochs"])
        use_tqdm = kwargs.pop("use_tqdm", True)

        for model, optimizer, config, v_p in zip(
            exp_bar, self.optimizers, self.all_configs, self.all_varying_params
        ):
            print(kwargs)
            exp_bar.set_description(f"{v_p} ")
            train_results.append(
                train_community(
                    model,
                    optimizer,
                    config,
                    loaders,
                    stop_acc=stop_acc,
                    device=device,
                    n_epochs=n_epochs,
                    use_tqdm=use_tqdm,
                    pbar=exp_bar,
                    **kwargs,
                )
            )

        self.results["train_results"] = train_results
        self.results["state_dicts"] = [model.state_dict() for model in self._models]
        if save:
            self.save_result_df()

    def compute_retraining_metric(
        self, loaders, save=False, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        retrain_results = []
        tqdm_f = tqdm_n if is_notebook() else tqdm
        exp_bar = tqdm_f(self.models)
        for model, config, v_p in zip(
            exp_bar, self.all_configs, self.all_varying_params
        ):
            exp_bar.set_description(f"{v_p} ")
            retrain_results.append(
                compute_retraining_metric(
                    model,
                    config,
                    loaders,
                    device,
                    use_tqdm=True,
                    pbar=exp_bar,
                )
            )

        retraining_metrics, retraining_models, retraining_configs = list(
            zip(*retrain_results)
        )
        retrain_accs = [r["test_accs"][-1] for r in retraining_metrics]
        retrain_all_accs = [r["all_accs"][-1] for r in retraining_metrics]
        self.results["retrain_accs"] = retrain_accs
        self.results["retrain_all_accs"] = retrain_all_accs
        self.results["retrain_models"] = [m.state_dict() for m in retraining_models]
        self.results["retrain_configs"] = retraining_configs
        self._retrained_models = retraining_models
        if save:
            self.save_result_df()

    @property
    def all_configs(self):
        return [
            copy_and_change_config(self.default_config, v_p)
            for v_p in self.all_varying_params
        ]

    @property
    def models(self):
        if hasattr(self, "_models"):
            return self._models
        else:
            assert (
                hasattr(self, "results")
                and hasattr(self.results, "columns")
                and ("state_dicts" in self.results.columns)
            ), "No models found, run experiment first"
            self._models = []
            for state_dict, config in zip(
                self.results["state_dicts"].values, self.all_configs
            ):
                model, _ = init_model(config)
                model.load_state_dict(state_dict)
                self._models.append(model)
            return self._models

    @property
    def retrained_models(self):
        if hasattr(self, "_retrained_models"):
            return self._retrained_models
        else:
            assert (
                hasattr(self, "results")
                and hasattr(self.results, "columns")
                and ("retrain_models" in self.results.columns)
            ), "No models found, run experiment first"
            self._retrained_models = []
            for state_dict, config in zip(
                self.results["retrain_models"].values, self.results["retrain_configs"]
            ):
                model, _ = init_model(config)
                model.load_state_dict(state_dict)
                self._retrained_models.append(model)

    def __repr__(self) -> str:
        return f"Experiment with {len(self.models)} models, and vaying params: {self.varying_params}"
