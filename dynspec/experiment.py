import torch
import pandas as pd
from tqdm.notebook import tqdm as tqdm_n
from tqdm import tqdm
from itertools import product
import copy
from hashlib import sha1
from dynspec.models import init_model
from dynspec.training import train_community
from dynspec.retraining import (
    compute_retraining_metric,
    create_retraining_model,
    compute_random_timing_dynamics,
    compute_ablations_metric,
)
from dynspec.correlations import compute_correlation_metric


def find_and_change(config, param_name, param_value):
    """
    Recursively searches through a dictionary `config` for a key `param_name` and changes its value to `param_value`.

    Args:
        config (dict): The dictionary to search through.
        param_name (str): The name of the parameter to search for.
        param_value (any): The new value to set the parameter to.

    Returns:
        dict: The modified dictionary.
    """
    for key, value in config.items():
        if type(value) is dict:
            find_and_change(value, param_name, param_value)
        else:
            if key == param_name:
                config[key] = param_value

    return config


def copy_and_change_config(config, varying_params):
    """
    Creates a deep copy of the given configuration and changes the values of the specified parameters.

    Args:
        config (dict): The configuration to copy and modify.
        varying_params (dict): A dictionary of parameter names and their new values.

    Returns:
        dict: The modified configuration.
    """
    config = copy.deepcopy(config)
    for n, v in varying_params.items():
        find_and_change(config, n, v)

    return config


def get_all_v_params(varying_params, excluded_params={}):
    """
    Returns a list of dictionaries, where each dictionary contains a combination of varying parameters.

    Args:
    - varying_params (dict): A dictionary where each key is a parameter name and each value is a list of parameter values.
    - excluded_params (dict): A dictionary where each key is a parameter name that should be excluded from the output.

    Returns:
    - A list of dictionaries, where each dictionary contains a combination of varying parameters.
    """
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
    """
    A class for running experiments with varying parameters and configurations.

    Args:
        default_config (dict): The default configuration for the experiment.
        varying_params (dict): A dictionary of varying parameters and their possible values.
        load_save (bool): Whether to load previously saved results or create new ones.
        device (str, optional): The device to run the experiment on. Defaults to "cuda" if available, else "cpu".
        n_tests (int, optional): The number of tests to run for each combination of varying parameters. Defaults to 1.
        hash_type (int, optional): The type of hash to use for saving and loading results. Defaults to 2.

    Attributes:
        varying_params (dict): A dictionary of varying parameters and their possible values.
        all_varying_params (list): A list of all possible combinations of varying parameters.
        default_config (dict): The default configuration for the experiment.
        _models (list): A list of all models used in the experiment.
        optimizers (list): A list of all optimizers used in the experiment.
        hash_type (int): The type of hash used for saving and loading results.
        results (DataFrame): A DataFrame containing the results of the experiment.
        result_path (str): The path to the file where the results are saved.

    Methods:
        load_result_df(): Loads the results of the experiment from a file.
        save_result_df(): Saves the results of the experiment to a file.
        run(): Runs the experiment.
        compute_retraining(): Computes the retraining metric for the experiment.
        compute_random_timing(): Computes the random timing metric for the experiment.
        compute_correlations(): Computes the correlations metric for the experiment.
        compute_ablations(): Computes the ablations metric for the experiment.

    Properties:
        all_configs (list): A list of all possible configurations for the experiment.
        models (list): A list of all models used in the experiment.
        retrained_models (list): A list of all retrained models used in the experiment.
    """

    def __init__(
        self,
        default_config,
        varying_params,
        load_save,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_tests=1,
        hash_type=2,
    ) -> None:
        self.varying_params = varying_params
        self.all_varying_params = get_all_v_params(varying_params)
        self.default_config = default_config
        self._models, self.optimizers, self.schedulers = [], [], []
        self.hash_type = hash_type

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

                    model, optimizer, scheduler = init_model(config, device)
                    self._models.append(model)
                    self.optimizers.append(optimizer)
                    self.schedulers.append(scheduler)
            self.results = pd.DataFrame.from_dict(self.results)

        else:
            for state_dict, config in zip(
                self.results["state_dicts"].values, self.all_configs
            ):
                model, optimizer, scheduler = init_model(config, device)
                model.load_state_dict(state_dict)
                self._models.append(model)
                self.optimizers.append(optimizer)
                self.schedulers.append(scheduler)

    def load_result_df(self):
        if self.hash_type == 0:
            v_hash = sha1(str(self.varying_params).encode("utf-8")).hexdigest()
        elif self.hash_type == 1:
            v_hash = sha1(
                (str(self.default_config) + str(self.varying_params)).encode("utf-8")
            ).hexdigest()
        elif self.hash_type == 2:
            saving_config = copy.deepcopy(self.default_config)
            saving_config["training"].pop("stop_acc", None)
            saving_config["training"].pop("n_epochs", None)
            saving_config.pop("optim", None)

            v_hash = sha1(
                (str(saving_config) + str(self.varying_params)).encode("utf-8")
            ).hexdigest()

        self.result_path = f"results/pre-loaded-examples/{v_hash}"
        print(self.result_path)
        try:
            results = pd.read_pickle(self.result_path)
            self.loaded = True
            print("Results loaded")
        except FileNotFoundError:
            results = {}
            self.loaded = False
            print("No results found, creating new dict")

        return results, self.result_path

    def save_result_df(self):
        if self.hash_type == 0:
            v_hash = sha1(str(self.varying_params).encode("utf-8")).hexdigest()
        elif self.hash_type == 1:
            v_hash = sha1(
                (str(self.default_config) + str(self.varying_params)).encode("utf-8")
            ).hexdigest()
        elif self.hash_type == 2:
            saving_config = copy.deepcopy(self.default_config)
            saving_config["training"].pop("stop_acc", None)
            saving_config["training"].pop("n_epochs", None)
            saving_config.pop("optim", None)
            v_hash = sha1(
                (str(saving_config) + str(self.varying_params)).encode("utf-8")
            ).hexdigest()

        self.result_path = f"results/pre-loaded-examples/{v_hash}"
        print(self.result_path)

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
        stop_acc = kwargs.pop(
            "stop_acc", self.default_config["training"].get("stop_acc", 0.95)
        )
        n_epochs = kwargs.pop(
            "n_epochs", self.default_config["training"].get("n_epochs", 30)
        )
        use_tqdm = kwargs.pop("use_tqdm", True)

        for model, optimizer, scheduler, config, v_p in zip(
            exp_bar,
            self.optimizers,
            self.schedulers,
            self.all_configs,
            self.all_varying_params,
        ):
            exp_bar.set_description(f"{v_p} ")
            train_results.append(
                train_community(
                    model,
                    optimizer,
                    config,
                    loaders,
                    scheduler=scheduler,
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

    def compute_retraining(
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

    def compute_random_timing(
        self, loaders, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        tqdm_f = tqdm_n if is_notebook() else tqdm

        random_timing_results = []
        for net, config in zip(tqdm_f(self.retrained_models), self.all_configs):
            random_timing_results.append(
                compute_random_timing_dynamics(net, loaders, config, device)
            )
        self.results["random_timing"] = random_timing_results

    def compute_correlations(
        self, loaders, save=False, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        correlations_results = []
        tqdm_f = tqdm_n if is_notebook() else tqdm
        exp_bar = tqdm_f(self.models)
        for model, config, v_p in zip(
            exp_bar, self.all_configs, self.all_varying_params
        ):
            exp_bar.set_description(f"{v_p} ")
            correlations_results.append(
                compute_correlation_metric(
                    model,
                    loaders[1],
                    config,
                    device,
                    use_tqdm=True,
                    pbar=exp_bar,
                )
            )

        self.results["correlations"] = correlations_results
        if save:
            self.save_result_df()

    def compute_ablations(
        self, loaders, save=False, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        ablations_results = []
        tqdm_f = tqdm_n if is_notebook() else tqdm
        exp_bar = tqdm_f(self.retrained_models, desc="Ablations Metric: ")
        for model, config, v_p in zip(
            exp_bar, self.all_configs, self.all_varying_params
        ):
            exp_bar.set_description(f"Ablation Metric, {v_p} ")
            exp_bar.set_description(f"{v_p} ")
            ablations_results.append(
                compute_ablations_metric(
                    model,
                    config,
                    loaders,
                    device,
                )
            )

        ablations_results, ablated_models, ablations_configs = list(
            zip(*ablations_results)
        )
        all_ablations_results = []
        for res, model, config in zip(
            ablations_results, ablated_models, ablations_configs
        ):
            all_ablations_results.append({})
            all_ablations_results[-1]["ablations_accs"] = res["test_acc"]
            all_ablations_results[-1]["ablated_models"] = model.state_dict()
            all_ablations_results[-1]["ablations_configs"] = config

        self.results["ablations"] = all_ablations_results
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
            if not (
                hasattr(self, "results")
                and hasattr(self.results, "columns")
                and ("retrain_models" in self.results.columns)
            ):
                print("No models found, run experiment first")
                return None
            else:
                _retrained_models = []
                for model, state_dict, config in zip(
                    self.models, self.results["retrain_models"].values, self.all_configs
                ):
                    r_model, _ = create_retraining_model(model, config)
                    r_model.load_state_dict(state_dict)
                    _retrained_models.append(r_model)
                self._retrained_models = _retrained_models
                return self._retrained_models

    def __repr__(self) -> str:
        return f"Experiment with {len(self.models)} models, and vaying params: {self.varying_params}"
