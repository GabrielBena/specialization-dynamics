import torch
import torch.nn as nn
import numpy as np

from dynspec.training import train_community
from dynspec.models import Readout, init_model
from dynspec.datasets import get_datasets

metric_norm = lambda m: np.clip(m - 0.1 / (1 - 0.1), 0, 1)
diff_metric = lambda metric: (metric[0] - metric[1]) / ((metric[0]) + (metric[1]))
global_diff_metric = (
    lambda metric: np.abs(
        diff_metric(metric_norm(metric[0])) - diff_metric(metric_norm(metric[1]))
    )
    / 2
)


def readout_mask(n_modules, n_in, n_out, ag_to_mask=None):
    """
    Create a mask for the readout layer when retraining a model, to only let pass input from a single module

    Args:
        n_modules (int): number of modules in the model
        n_in (int): number of input features
        n_out (int):  number of output features
        ag_to_mask (int, optional): module to mask during retraining. If None, both modules are used. Defaults to None.

    Returns:
        _type_: _description_
    """
    mask = torch.ones(n_modules).unsqueeze(0)
    if ag_to_mask is not None:
        mask[:, ag_to_mask] = 0
    mask = mask.repeat_interleave(n_out, 0).repeat_interleave(n_in, 1)
    # plt.imshow(mask)
    return mask


def compute_retraining_metric(
    original_model,
    config,
    loaders,
    device=torch.device("cuda"),
    use_tqdm=False,
    **kwargs,
):
    # model = copy.deepcopy(original_model)

    state_dict_copy = original_model.state_dict().copy()
    model, _ = init_model(config, device=device)
    model.load_state_dict(state_dict_copy)
    retraining_config = config.copy()

    n_classes = config["data"]["n_classes_per_digit"]
    nb_steps = config["data"]["nb_steps"]
    n_modules = config["modules"]["n_modules"]
    ag_hidden_size = config["modules"]["hidden_size"]
    n_targets = 2

    retraining_config["readout"]["output_size"] = [
        [n_classes for _ in range(n_targets)] for _ in range(n_modules + 1)
    ]
    n_hid = 30
    retraining_config["readout"]["n_hid"] = [
        [n_hid for _ in range(n_targets)] for _ in range(n_modules + 1)
    ]
    retraining_config["readout"]["common_readout"] = True
    retraining_config["readout"]["retraining"] = True

    out_masks = torch.stack(
        [
            torch.stack(
                [
                    readout_mask(
                        model.n_modules,
                        model.hidden_size,
                        n_hid,
                        ag_to_mask=i,
                    )
                    if n_hid is not None
                    else readout_mask(
                        model.n_modules,
                        model.hidden_size,
                        n_classes,
                        ag_to_mask=i,
                    )
                    for i in list(range(model.n_modules))[::-1] + [None]
                ]
            )
            for _ in range(n_targets)
        ]
    )
    model.readout = Readout(
        retraining_config["readout"],
        n_modules,
        ag_hidden_size,
        out_masks=out_masks,
    )

    for n, p in model.named_parameters():
        if p.requires_grad and not "readout" in n:
            p.requires_grad = False

    model.to(device)
    retraining_config["decision"] = ("none", "none")

    retraining_config["training"]["task"] = [
        [[str(i) for i in range(n_targets)] for _ in range(n_modules + 1)]
        for _ in range(nb_steps)
    ]

    retraining_config["training"]["n_epochs"] = 3

    retraining_optimizer = torch.optim.AdamW(
        model.parameters(), **retraining_config["optim"]
    )

    retraining_results = train_community(
        model,
        retraining_optimizer,
        retraining_config,
        loaders,
        stop_acc=0.95,
        device=device,
        show_all_acc=-1,
        use_tqdm=use_tqdm,
        **kwargs,
    )

    return retraining_results, model, retraining_config


def compute_ablations_metric():
    return


if __name__ == "__main__":
    task = ["parity-digits", "inv-parity-digits"]

    modules_config = {
        "n_modules": 2,
        "hidden_size": 10,  # will be changed later
        "n_layers": 1,
        "dropout": 0.0,
        "cell_type": str(nn.RNN),
    }

    connections_config = {"sparsity": 1}  # Will be changed later

    n_outs = {
        "none": [10, 10],
        "parity-digits": 10,
        "inv-parity-digits": 10,
        "parity-digits-both": [10, 10],
        "parity-digits-sum": 2,
        "sum": 20,
        "bitxor": 16,
        "bitxor-last-1": 2,
        "1": 10,
        "0": 10,
        "inv": 10,
    }

    input_config = {"input_size": 784, "common_input": False}
    optim_config = {"lr": 1e-3, "weight_decay": 1e-5}

    readout_config = {"common_readout": False, "n_hid": 5}

    decision = ["last", "max"]

    training_config = {"n_epochs": 30, "task": task, "check_grad": False}

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_modules = 2
    n_classes_per_digit = 10
    n_classes = n_classes_per_digit * n_modules
    nb_steps = 5

    data_config = {
        # ------ Change if needed------
        "batch_size": 256,
        "input_size": 28,
        "use_cuda": use_cuda,
        "data_type": "double_digits",
        "n_digits": n_modules,
        "n_classes": n_classes,
        "n_classes_per_digit": n_classes_per_digit,
        "nb_steps": nb_steps,
        # cov ratio : controls the probabilty of seeing D1 == D2, default = 1 (chance probability)
        "cov_ratio": 1,
        # noise ratio : controls the ammount of noise added to the input , default = 0.5
        "noise_ratio": 0.4,
        # random start : add stochasticity by having input start at random times from pure noise, default = False
        "random_start": False,
        # --------------------------
    }
    all_data = get_datasets("../data/", data_config)
    datasets, loaders = all_data[data_config["data_type"]]
    len(datasets[0])

    default_config = {
        "modules": modules_config,
        "connections": connections_config,
        "input": input_config,
        "readout": readout_config,
        "data": data_config,
        "decision": decision,
        "training": training_config,
        "optim": optim_config,
    }

    model, optimizer = init_model(default_config)
    model(input=torch.randn(5, 128, 1568))
    print(model)

    compute_retraining_metric(model, default_config, loaders, use_tqdm=True)
