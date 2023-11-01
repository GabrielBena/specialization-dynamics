import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dynspec.metrics import global_diff_metric, diff_metric
import pandas as pd
from matplotlib.colors import LogNorm


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


def plot_random_timings(experiment):
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
        test = 0
        u_masks, accs, start_times = (
            results["all_u_masks"],
            results["all_accs"],
            results["all_start_times"],
        )
        nb_steps = experiment.default_config["data"]["nb_steps"]
        n_modules = experiment.default_config["modules"]["n_modules"]
        for mask, pair in zip(u_masks, start_times.unique(dim=0)):
            for step, accs_step in enumerate(accs[test]):
                for ag, accs_ag in enumerate(accs_step):
                    # plot_data['global_metric'].append(diff_metric(all_accs[step, -1, :, mask].mean(0)))
                    plot_data["local_metric"].append(
                        diff_metric(accs_ag[:, mask].mean(1))
                    )
                    # plot_data['local_metric_1'].append(diff_metric(all_accs[step, 1, :, mask].mean(0)))
                    plot_data["t0"].append(pair[0].item())
                    plot_data["t1"].append(pair[1].item())
                    plot_data["t0_t1"].append(tuple(pair.cpu().data.numpy()))
                    plot_data["step"].append(step)
                    plot_data["ag"].append(ag)
                    for k, v in vp.items():
                        plot_data.setdefault(k, [])
                        plot_data[k].append(v)
                    # plot_data['x'].append([-1, 1][ag] * (accs.shape[0] - step))
                    # plot_data['x'].append(step)

            # plot_data['local_metric'].append(diff_metric(accs[step, -1, :, mask].mean(0)))
            # plot_data['t0'].append(pair[0].item())
            # plot_data['t1'].append(pair[1].item())
            # plot_data['t0_t1'].append(tuple(pair.cpu().data.numpy()))
            # plot_data['step'].append(nb_steps)
            # plot_data['ag'].append(-1)

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
        # ax.vlines(x = [t0 - 0.5 ], ymin = -1, ymax = 1, color = 'blue', alpha = .2)
        # ax.vlines(x = [t1 - 0.5 ], ymin = -1, ymax = 1, color = 'red', alpha = .2)
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
