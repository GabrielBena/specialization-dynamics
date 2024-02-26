import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
import warnings

from dynspec.retraining import global_diff_metric, diff_metric, metric_norm_acc
from dynspec.data_process import process_data
from dynspec.tasks import get_task_target
from dynspec.decision import get_decision
from dynspec.training import get_acc


def set_style():
    """
    Sets the plotting style using a style sheet file located in the same directory as this script.
    """
    file_path = os.path.realpath(__file__)
    file_path = file_path.replace("plotting.py", "style_sheet.mplstyle")
    plt.style.use(file_path)


def single_filter(data, key, value):
    """
    Filter a pandas DataFrame based on a single key-value pair.

    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame to filter.
    key : str
        The column name to filter on. If the key starts with "!", the filter will exclude rows where the column value matches `value`.
    value : any
        The value to filter on. If None, the filter will return rows where the column value is null.

    Returns:
    --------
    pandas.Series
        A boolean Series indicating which rows of `data` pass the filter.
    """
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


def filter_data(data: pd.DataFrame, v_params: dict) -> tuple:
    """
    Filters the input data based on the given parameters.

    Args:
        data (pd.DataFrame): The input data to be filtered.
        v_params (dict): A dictionary of filter parameters.

    Returns:
        tuple: A tuple containing the filtered data and a boolean mask indicating which rows were filtered out.
    """
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
    """
    Plots the recurrent and connections masks for each model in the given experiment.

    Args:
        experiment (Experiment): The experiment object containing the models to plot.
        plot_input (bool, optional): Whether to plot the input masks as well. Defaults to False.
    """
    n1, n2 = int(np.sqrt(len(experiment.all_configs))), int(
        np.ceil(np.sqrt(len(experiment.all_configs)))
    )
    if n1 * n2 < len(experiment.models):
        n2 += 1

    fig = plt.figure(
        figsize=(n2 * 2, n1 * 2 * (1 + plot_input * 0.3)), constrained_layout=True
    )

    subfigs = fig.subfigures(1 + plot_input, height_ratios=[1, 0.3][: (1 + plot_input)])
    if not plot_input:
        subfigs = np.array([subfigs])

    subfigs[0].suptitle("Recurrent + Connections Masks")
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


def plot_accs(experiment):
    """
    Plots the test accuracies for all training runs in the given experiment.

    Args:
        experiment: An instance of the Experiment class.
    """

    set_style()
    train_results, all_varying_params = (
        experiment.results["train_results"],
        experiment.results["varying_params"],
    )
    n1, n2 = int(np.sqrt(len(train_results))), int(np.ceil(np.sqrt(len(train_results))))

    if n1 * n2 < len(train_results):
        n2 += 1

    fig, axs = plt.subplots(
        n1,
        n2,
        figsize=(2.5 * n2, n1 * 1.5),
        constrained_layout=True,
        sharey=True,
        sharex=True,
    )
    if n1 == n2 == 1:
        axs = np.array([axs])

    for vp, t, ax in zip(all_varying_params, train_results, axs.flatten()):
        ax.plot(t["test_accs"].reshape(t["test_accs"].shape[0], -1))

        legend = {"hidden_size": "n", "sparsity": "p"}
        vp_legend = ", ".join([f"{legend[k]} = {v}" for k, v in vp.items()])
        ax.set_title(vp_legend)


def plot_metric_results(experiment):
    """
    Plots the metric results of an experiment.

    Args:
    - experiment: an Experiment object containing the results to plot.

    Returns:
    - metric_global_data: a pandas DataFrame containing the metric data.
    """
    set_style()
    warnings.filterwarnings("ignore")
    metric_global_data = {"step": []}
    if "retrain_accs" in experiment.results.columns:
        for r_accs, vp in zip(
            experiment.results["retrain_accs"], experiment.all_varying_params
        ):
            metric_global_data.setdefault("metric", [])
            metric_global_data.setdefault("metric_name", [])
            metric_global_data["metric"].extend(
                global_diff_metric(metric_norm_acc(r)) for r in r_accs
            )
            metric_global_data["metric_name"].extend(["retraining"] * len(r_accs))
            for k, v in vp.items():
                metric_global_data.setdefault(k, [])
                metric_global_data[k].extend([v] * len(r_accs))

            metric_global_data["step"].extend(range(len(r_accs)))

    if "correlations" in experiment.results.columns:
        for correlations, vp in zip(
            experiment.results["correlations"], experiment.all_varying_params
        ):
            norm_correlations = correlations["norm_correlations"]
            metric_global_data.setdefault("metric", [])
            metric_global_data.setdefault("metric_name", [])
            metric_global_data["metric"].extend(
                global_diff_metric(c) for c in norm_correlations
            )
            metric_global_data["metric_name"].extend(
                ["correlation"] * len(norm_correlations)
            )
            for k, v in vp.items():
                metric_global_data.setdefault(k, [])
                metric_global_data[k].extend([v] * len(norm_correlations))
            metric_global_data["step"].extend(range(len(norm_correlations)))

    if "ablations" in experiment.results.columns:
        for ablations, vp in zip(
            experiment.results["ablations"], experiment.all_varying_params
        ):
            ablations_accs = ablations["ablations_accs"]
            metric_global_data.setdefault("metric", [])
            metric_global_data.setdefault("metric_name", [])
            metric_global_data["metric"].extend(
                global_diff_metric(metric_norm_acc(a)) for a in ablations_accs
            )
            metric_global_data["metric_name"].extend(
                ["ablations"] * len(ablations_accs)
            )
            for k, v in vp.items():
                metric_global_data.setdefault(k, [])
                metric_global_data[k].extend([v] * len(ablations_accs))
            metric_global_data["step"].extend(range(len(ablations_accs)))

    metric_global_data = pd.DataFrame(metric_global_data)
    metric_global_data["sparsity_"] = metric_global_data.apply(
        lambda x: x.sparsity if x.sparsity > 0 else 1 / x.hidden_size**2, axis=1
    )
    metric_global_data["q_metric"] = metric_global_data["sparsity_"].apply(
        lambda x: 0.5 * (1 - x) / (1 + x)
    )

    metrics = metric_global_data["metric_name"].unique()
    fig, axs = plt.subplots(
        2,
        len(metrics),
        figsize=(4 * len(metrics), 6),
        # sharey=True,
        constrained_layout=True,
    )

    norm = Normalize(
        vmin=metric_global_data["hidden_size"].min(),
        vmax=metric_global_data["hidden_size"].max(),
    )
    colors = sns.palettes.color_palette("viridis", len(metrics))
    for m, (metric, axs_m) in enumerate(zip(metrics, axs.T)):
        for j, (x, x_label, ax) in enumerate(
            zip(
                ["sparsity_", "q_metric"],
                [
                    "Sparsity (p) of interconnections",
                    "Structural Q Metric of the model",
                ],
                axs_m,
            )
        ):
            ln = sns.lineplot(
                data=filter_data(
                    metric_global_data,
                    {
                        "step": experiment.default_config["data"]["nb_steps"] - 1,
                        "metric_name": metric,
                    },
                )[0],
                x=x,
                y="metric",
                palette=(
                    "viridis"
                    if len(metric_global_data["hidden_size"].unique()) > 1
                    else None
                ),
                hue=(
                    "hidden_size"
                    if len(metric_global_data["hidden_size"].unique()) > 1
                    else None
                ),
                color=(
                    colors[m]
                    if len(metric_global_data["hidden_size"].unique()) == 1
                    else None
                ),
                ax=ax,
            )
            if x == "sparsity_":
                ax.set_xscale("log")
            # plt.yscale("log")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Model Specialization")
            if x == "sparsity_":
                ax.set_title(metric)
            # if (len(metric_global_data["hidden_size"].unique()) == 1) or (
            #     # m * len(metrics) + j != 2 * len(metrics) - 1
            # ):
            if len(metric_global_data["hidden_size"].unique()) > 1:
                ax.legend().remove()

            cm = plt.cm.get_cmap("viridis")
            sm = plt.cm.ScalarMappable(
                cmap=cm,
                norm=norm,
            )
            if (
                m == len(metrics) - 1
                and len(metric_global_data["hidden_size"].unique()) > 1
            ):
                cbar = fig.colorbar(sm, ax=ax, label="Hidden Size (n)")
                cbar.set_ticks(metric_global_data["hidden_size"].unique())
                cbar.set_ticklabels(metric_global_data["hidden_size"].unique())

    return metric_global_data


def plot_random_timings(experiment):
    """
    Plots the local metric for each module in the experiment, as a function of time step, for different sparsity levels.
    The function returns a pandas DataFrame containing the data used for the plot.

    Args:
        experiment: An instance of the Experiment class containing the results to be plotted.

    Returns:
        A pandas DataFrame containing the data used for the plot.
    """
    plot_data = {
        "t0": [],
        "t1": [],
        "t0_t1": [],
        "step": [],
        "local_metric": [],
        "ag": [],
    }

    for vp, results in zip(
        experiment.all_varying_params, experiment.results["random_timing"]
    ):
        u_masks, accs, start_times = (
            results["all_u_masks"],
            results["all_accs"],
            results["all_start_times"],
        )
        nb_steps = experiment.default_config["data"]["nb_steps"]
        n_modules = experiment.default_config["modules"]["n_modules"]
        for mask, pair in zip(u_masks, start_times.unique(dim=0)):
            for step, accs_step in enumerate(accs[0]):
                for ag, accs_ag in enumerate(accs_step):
                    # plot_data['global_metric'].append(diff_metric(all_accs[step, -1, :, mask].mean(0)))
                    mean_accs = accs_ag[:, mask].mean(1)
                    plot_data["local_metric"].append(diff_metric(mean_accs))
                    # plot_data['local_metric_1'].append(diff_metric(all_accs[step, 1, :, mask].mean(0)))
                    plot_data["t0"].append(pair[0].item())
                    plot_data["t1"].append(pair[1].item())
                    plot_data["t0_t1"].append(tuple(pair.cpu().data.numpy()))
                    plot_data["step"].append(step)
                    plot_data["ag"].append(ag)
                    for k, v in vp.items():
                        plot_data.setdefault(k, [])
                        plot_data[k].append(v)

    plot_data = pd.DataFrame.from_dict(plot_data)
    last_ts_data = [
        filter_data(plot_data, {"ag": 2, "step": nb_steps - 1})[0]
        for _ in range(n_modules)
    ]
    for ag, data in enumerate(last_ts_data):
        data.loc[:, "ag"] = ag
        data.loc[:, "step"] = nb_steps

    last_ts_data = pd.concat(last_ts_data)
    plot_data = filter_data(pd.concat([plot_data, last_ts_data]), {"!ag": 2})[0]
    # plot_data = plot_data.loc[~plot_data["local_metric"].isna()]

    fig, axs = plt.subplots(
        3,
        3,
        figsize=(3 * 3.5, 3 * 2),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    set_style()

    for u, ax in zip(plot_data["t0_t1"].unique(), axs.T.flatten()):
        t0, t1 = u
        sns.lineplot(
            filter_data(
                plot_data,
                {
                    "t0_t1": u,
                },  # "sparsity": [sparsities[i] for i in [0, 1, 2, 3, 4]]},
            )[0],
            y="local_metric",
            x="step",
            ax=ax,
            style="ag",
            hue="sparsity",
            palette="viridis",
            hue_norm=LogNorm(),
        )

        c1 = ax.fill_betweenx([0.1, 1], 0, nb_steps, alpha=0.5, label="M0")
        c2 = ax.fill_betweenx([-0.1, -1], 0, nb_steps, alpha=0.5, label="M1")

        ax.arrow(
            t0 - 0.5,
            0,
            0,
            1,
            alpha=0.5,
            width=0.1,
            head_width=0.2,
            facecolor=c1.get_facecolor()[0],
            edgecolor="black",
            length_includes_head=True,
            linewidth=1,
        )
        ax.arrow(
            t1 - 0.5,
            0,
            0,
            -1,
            alpha=0.5,
            width=0.1,
            head_width=0.2,
            facecolor=c2.get_facecolor()[0],
            edgecolor="black",
            length_includes_head=True,
            linewidth=1,
        )
        ax.legend()
        # ax.set_title(u)

        if t1 == 3:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("")

        if t0 == 1:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("")

        ax.legend().remove()

        ax.grid(True, alpha=0.2, linestyle="dashed", linewidth=2, color="grey")
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

    plt.show()
    return plot_data


def plot_model_pair_behavior(experiment, loaders, annot=True):

    result_dict = {}
    for vp, model, config in zip(
        experiment.all_varying_params, experiment.models, experiment.all_configs
    ):
        n_classes = config["data"]["n_classes_per_digit"]

        result_dict[str(vp)] = {}

        for key in ["acc", "decision", "target", "task_target"] + list(
            experiment.varying_params.keys()
        ):
            result_dict[str(vp)][key] = []

        device = list(model.parameters())[0].device

        for data, target in loaders[1]:
            data, _ = process_data(data, config["data"])
            data, target = data.to(device), target.to(device)
            t_target = get_task_target(
                target,
                config["training"]["task"],
                n_classes=n_classes,
            )
            out = model(data)[0]
            out, deciding_ag = get_decision(out, *config["decision"])
            acc, all_accs = get_acc(out, t_target, config["decision"][1] == "both")

            result_dict[str(vp)]["decision"].extend(
                deciding_ag.cpu().data.numpy().tolist()
            )
            result_dict[str(vp)]["acc"].extend(all_accs.tolist())
            result_dict[str(vp)]["target"].extend(target.cpu().data.numpy().tolist())
            result_dict[str(vp)]["task_target"].extend(
                t_target.cpu().data.numpy().tolist()
            )
            for k, v in vp.items():
                result_dict[str(vp)][k].append(v)

    result_dict = {
        vp: {k: np.array(v) for k, v in r.items()} for vp, r in result_dict.items()
    }

    fig = plt.figure(
        figsize=(3 * n_classes, n_classes * len(result_dict)), constrained_layout=True
    )
    subfigs = fig.subfigures(len(result_dict), 1)
    if len(result_dict) == 1:
        subfigs = np.array([subfigs])
    for (vp, res), subfig in zip(result_dict.items(), subfigs):

        axs = subfig.subplots(1, 3)
        subfig.suptitle(str(vp))

        plot_dict = {
            k: np.full((n_classes, n_classes), None, dtype=object)
            for k in ["acc", "decision", "task_target"]
        }

        for pair in np.unique(res["target"], axis=0):
            mask = np.all(res["target"] == pair, axis=1)
            for k in plot_dict.keys():
                plot_dict[k][pair[0], pair[1]] = res[k][mask].mean()

        for ax, k in zip(axs, plot_dict.keys()):
            sns.heatmap(
                np.round(plot_dict[k].astype(float), 2), annot=annot, ax=ax, cbar=False
            )
            ax.set_title(k)
