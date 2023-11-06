import torch
import torch.nn as nn
import numpy as np

from dynspec.training import train_community, test_community, is_notebook, get_acc
from dynspec.models import Readout, init_model
from dynspec.data_process import process_data
from tqdm.notebook import tqdm as tqdm_n
from tqdm.notebook import tqdm
import copy

metric_norm_acc = lambda m: np.clip(m - 0.1 / (1 - 0.1), 1e-5, 1)
diff_metric = lambda metric: (metric[0] - metric[1]) / ((metric[0]) + (metric[1]))
global_diff_metric = (
    lambda metric: np.abs(diff_metric(metric[0]) - diff_metric(metric[1])) / 2
)


def reccursive_stack(input):
    """
    Recursively stacks a list of PyTorch tensors into a single tensor.

    Args:
        input (list): A list of PyTorch tensors.

    Returns:
        torch.Tensor: A stacked PyTorch tensor.
    """
    try:
        return torch.stack(input)
    except TypeError:
        return torch.stack([reccursive_stack(i) for i in input])


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


def create_retraining_model(
    original_model,
    config,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
):
    """
    Creates a new model for retraining based on the original model and configuration.

    Args:
        original_model (nn.Module): The original model to be retrained.
        config (dict): The configuration dictionary of the original model
        device (torch.device, optional): The device to use for the new model. Defaults to "cuda" if available, else "cpu".

    Returns:
        nn.Module: The new model for retraining.
        dict: The configuration dictionary for the new model.
    """

    state_dict_copy = original_model.state_dict().copy()
    model, _ = init_model(config, device=device)
    model.load_state_dict(state_dict_copy)
    retraining_config = config.copy()

    n_classes = config["data"]["n_classes_per_digit"]
    n_modules = config["modules"]["n_modules"]
    ag_hidden_size = config["modules"]["hidden_size"]
    n_targets = 2

    retraining_config["readout"]["output_size"] = [
        [n_classes for _ in range(n_modules + 1)] for _ in range(n_targets)
    ]
    n_hid = 30
    retraining_config["readout"]["n_hid"] = [
        [n_hid for _ in range(n_modules + 1)] for _ in range(n_targets)
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
    return model, retraining_config


def compute_retraining_metric(
    original_model,
    config,
    loaders,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    use_tqdm=False,
    **kwargs,
):
    """
    Computes the retraining metric for a given original model, configuration, and data loaders.

    Args:
        original_model (nn.Module): The original model to be retrained.
        config (dict): The configuration dictionary.
        loaders (dict): The data loaders.
        device (torch.device, optional): The device to use for computation. Defaults to "cuda" if available, else "cpu".
        use_tqdm (bool, optional): Whether to use tqdm for progress bars. Defaults to False.
        **kwargs: Additional keyword arguments to be passed to the train_community function.

    Returns:
        tuple: A tuple containing the retraining results, the retrained model, and the retraining configuration.
    """
    nb_steps = config["data"]["nb_steps"]
    n_modules = config["modules"]["n_modules"]
    n_targets = 2

    model, retraining_config = create_retraining_model(
        original_model, config, device=device
    )

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


def compute_ablations_metric(
    retrained_model,
    config,
    loaders,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
):
    """
    Computes the ablations metric for a retrained model.

    Args:
        retrained_model (nn.Module): The retrained model to compute the ablations metric for.
        config (dict): The configuration dictionary for the model.
        loaders (list): data loaders.
        device (torch.device, optional): The device to use for computation. Defaults to "cuda" if available, else "cpu".

    Returns:
        tuple: A tuple containing the ablations results, the model, and the ablations configuration.
    """
    # function code here

    model = copy.deepcopy(retrained_model)
    model.readout.layers = nn.ModuleList(
        [
            nn.ModuleList([copy.deepcopy(r[-1]) for _ in range(3)])
            for r in model.readout.layers
        ]
    )
    ablations_config = config.copy()
    ablations_config["decision"] = ["none", "none"]
    ablations_config["training"]["task"] = [
        [[str(i) for i in range(2)] for _ in range(3)]
        for _ in range(config["data"]["nb_steps"])
    ]
    for readout in model.readout.layers:
        for ag_to_mask, masked_r in zip([0, 1, None], readout):
            if ag_to_mask is None:
                continue
            masked_r[0].parametrizations.weight[0].mask[
                :, ag_to_mask * model.hidden_size : (ag_to_mask + 1) * model.hidden_size
            ] = 0
    ablations_results = test_community(model, device, loaders[1], ablations_config)
    return ablations_results, model, ablations_config


def compute_random_timing_dynamics(model, loaders, config, device):
    """
    Computes random timing dynamics for a given model and dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        loaders (tuple): A tuple of PyTorch DataLoader objects for the train and test sets.
        config (dict): A dictionary containing the configuration parameters for the experiment.
        device (str): The device to use for computations (e.g. "cpu" or "cuda").

    Returns:
        A dictionary containing the random timing dynamics.
    """
    random_config = config.copy()
    random_config["data"]["random_start"] = True

    nb_steps = config["data"]["nb_steps"]
    n_modules = config["modules"]["n_modules"]
    n_targets = 2

    all_outputs = []
    all_start_times = []
    all_targets = []

    for data, target in loaders[1]:
        data, start_times = process_data(data, random_config["data"])
        data, target = data.to(device), target.to(device)
        outputs, _ = model(data)
        outputs = reccursive_stack(outputs).transpose(0, 2).squeeze()
        all_outputs.append(outputs)  # steps x modules x target
        all_start_times.append(start_times)
        all_targets.append(target)

    all_outputs = torch.cat(all_outputs, -2).unsqueeze(0)
    all_start_times = torch.cat(all_start_times, -2)
    all_targets = torch.cat(all_targets, -2).unsqueeze(0)

    all_accs = np.stack(
        [
            get_acc(
                out,
                [
                    [[t for t in target.T] for _ in range(n_modules + 1)]
                    for _ in range(nb_steps)
                ],
            )[1]
            for out, target in zip(all_outputs, all_targets)
        ]
    )

    all_u_masks = [
        (all_start_times == u).all(-1) for u in all_start_times.unique(dim=0)
    ]

    return {
        "all_accs": all_accs,
        "all_u_masks": all_u_masks,
        "all_outputs": all_outputs,
        "all_targets": all_targets,
        "all_start_times": all_start_times,
    }
