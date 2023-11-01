import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dynspec.metrics import global_diff_metric
import pandas as pd


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


def plot_model_masks(experiment, plot_input=False):
    n1, n2 = int(np.sqrt(len(experiment.all_configs))), int(
        np.ceil(np.sqrt(len(experiment.all_configs)))
    )
    if n1 * n2 < len(experiment.models):
        n2 += 1

    fig = plt.figure(
        figsize=(n2 * 3, n1 * 3 * (1 + plot_input * 0.3)), constrained_layout=True
    )

    subfigs = fig.subfigures(1 + plot_input, height_ratios=[1, 0.3][: (1 + plot_input)])
    if not plot_input:
        subfigs = np.array([subfigs])

    subfigs[0].suptitle("Recurrent and Connection Masks")
    axs = subfigs[0].subplots(n1, n2)
    if len(experiment.models) == 1:
        axs = np.array([axs]).T

    for ax, model in zip(axs.flatten(), experiment.models):
        ax.imshow(
            (
                model.masks["comms_mask"][
                    : model.n_modules * model.modules_config["hidden_size"]
                ]
                + model.masks["rec_mask"][
                    : model.n_modules * model.modules_config["hidden_size"]
                ]
            )
            .cpu()
            .numpy(),
            vmin=0,
            vmax=1,
        )
        ax.set_title(
            f'n = {model.modules_config["hidden_size"]}, p = {model.connections_config["sparsity"]}'
        )

    if plot_input:
        subfigs[1].suptitle("Input Masks")
        axs = subfigs[1].subplots(n1, n2)
        if len(experiment.models) == 1:
            axs = np.array([axs]).T
        for ax, model in zip(axs, experiment.models):
            ax.imshow(
                (model.masks["input_mask"])
                .cpu()
                .numpy()[: model.n_modules * model.modules_config["hidden_size"], :],
                aspect="auto",
                vmin=0,
                vmax=1,
            )


def plot_accs(general_training_results):
    train_results, all_varying_params = (
        general_training_results["train_results"],
        general_training_results["varying_params"],
    )
    n1, n2 = int(np.sqrt(len(train_results))), int(np.ceil(np.sqrt(len(train_results))))

    if n1 * n2 < len(train_results):
        n2 += 1

    fig, axs = plt.subplots(
        n1,
        n2,
        figsize=(3 * n2, n1 * 1.5),
        constrained_layout=True,
        sharey=True,
        sharex=True,
    )
    if n1 == n2 == 1:
        axs = np.array([axs])

    for vp, t, ax in zip(all_varying_params, train_results, axs.flatten()):
        ax.plot(t["test_accs"].reshape(t["test_accs"].shape[0], -1))
        ax.set_title(vp)


def plot_retraining_results(experiment):
    retrain_global_data = {k: [] for k in ["metric", "step"]}
    for r_accs, vp in zip(
        experiment.results["retrain_accs"], experiment.all_varying_params
    ):
        retrain_global_data["metric"].extend(global_diff_metric(r) for r in r_accs)
        for k, v in vp.items():
            retrain_global_data.setdefault(k, [])
            retrain_global_data[k].extend([v] * len(r_accs))

        retrain_global_data["step"].extend(range(len(r_accs)))

    retrain_global_data = pd.DataFrame(retrain_global_data)
    retrain_global_data["sparsity_"] = retrain_global_data.apply(
        lambda x: x.sparsity if x.sparsity > 0 else 1 / x.hidden_size**2, axis=1
    )
    retrain_global_data["q_metric"] = retrain_global_data["sparsity_"].apply(
        lambda x: 0.5 * (1 - x) / (1 + x)
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for x, x_label, ax in zip(
        ["sparsity_", "q_metric"],
        ["Sparsity (p) of interconnections", "Structural Q Metric of the model"],
        axs,
    ):
        sns.lineplot(
            data=filter_data(
                retrain_global_data,
                {"step": experiment.default_config["data"]["nb_steps"] - 1},
            )[0],
            x=x,
            y="metric",
            palette="viridis",
            hue="hidden_size",
            ax=ax,
        )
        if x == "sparsity_":
            ax.set_xscale("log")
        # plt.yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Model Specialization")
