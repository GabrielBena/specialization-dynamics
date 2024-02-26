import torch
import numpy as np
from scipy.stats import pearsonr
from tqdm.notebook import tqdm as tqdm_n
from tqdm.notebook import tqdm
from dynspec.data_process import process_data
from dynspec.training import is_notebook
from cka.CKA import CKA


def fixed_information_data(
    data, target, fixed, fixed_mode="label", permute_other=True, n_modules=2
):
    """
    Create a modified version of the data, to be used for correlation metric.
    One of the quality is fixed (digit label, or parity, etc), for on of the input digit

    Args:
        data (torch.tensor): input data
        target (torch.tensor): input labels
        fixed (int): digit to be fixed
        fixed_mode (str, optional): quality to fix. Defaults to "label".
        permute_other (bool, optional): apply random permutation to the other (non-fixed) digit. Defaults to True.
        n_modules (int, optional): number of modules. Defaults to 2.

    Raises:
        NotImplementedError: fixed_mode not recognized

    Returns:
        new_data: list of modified data, with each presenting a fixed quality (size : n_quality x n_timestep x n_batch x n_features)
    """

    data = data.clone()
    digits = torch.split(target, 1, dim=-1)
    bs = digits[0].shape[0]

    classes = [d.unique() for d in digits]

    if len(data.shape) == 3:
        data = torch.stack(data.split(data.shape[-1] // 2, dim=-1), 1)
        reshape = True
    else:
        reshape = False

    if permute_other:
        data[:, 1 - fixed, ...] = data[:, 1 - fixed, torch.randperm(bs), ...]

    if fixed_mode == "label":
        d_idxs = [torch.where(digits[fixed] == d)[0] for d in classes[fixed]]
    elif fixed_mode == "parity":
        d_idxs = [torch.where(digits[fixed] % 2 == p)[0] for p in range(2)]
    else:
        raise NotImplementedError('Fixation mode not recognized ("label" or "parity")')

    datas = [[data[:, j, idx, :] for idx in d_idxs] for j in range(2)]

    new_data = [torch.stack([d1, d2], axis=1) for d1, d2 in zip(*datas)]

    if reshape:
        new_data = [d.transpose(1, -2).flatten(start_dim=-2) for d in new_data]

    return new_data


cka_fn = CKA().linear_CKA
v_pearsonr = np.vectorize(pearsonr, signature="(n1),(n2)->(),()")
v_cka = lambda states1, states2: np.stack(
    [cka_fn(s1, s2) for s1, s2 in zip(states1, states2)], 0
)


def randperm_no_fixed(n):
    """
    Generates a random permutation of integers from 0 to n-1, such that no integer is fixed in place.

    Args:
        n (int): The number of integers to permute.

    Returns:
        torch.Tensor: A 1-D tensor of size n containing the randomly permuted integers.
    """

    perm = torch.randperm(n)

    if (torch.arange(n) == perm).any() and n > 4:
        return randperm_no_fixed(n)
    else:
        return perm


def get_correlation(model, data, corr_func="pearsonr"):
    """
    Compute the self-correlation between of module's hidden states

    Args:
        model (nn.Module): modular network
        data (torch.tensor): data to be processed by the network

    Returns:
        np.array: correlation between hidden states (size : n_timestep x n_modules x n_batch)
    """
    states = model(data)[1]

    agent_states = states.split(model.hidden_size, -1)
    agent_states = [ag_s.cpu().data.numpy() for ag_s in agent_states]
    perm = randperm_no_fixed(agent_states[0].shape[1])
    if corr_func == "pearsonr":
        corr = np.stack(
            [v_pearsonr(ag_s, ag_s[:, perm])[0] for ag_s in agent_states], 1
        )
    elif corr_func == "cka":
        corr = np.stack([v_cka(ag_s, ag_s[:, perm]) for ag_s in agent_states], 1)
    return corr


def compute_correlation_metric(
    model,
    loader,
    config,
    device=torch.device("cuda") if torch.cuda.is_available else torch.device("cpu"),
    n_samples=10,
    use_tqdm=True,
    pbar=None,
):
    """
    Computes the correlation metric for a given model and data loader.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        loader (DataLoader): The data loader to use for evaluation.
        config (dict): A dictionary containing the configuration parameters.
        device (torch.device, optional): The device to use for computation. Defaults to "cuda" if available, else "cpu".
        n_samples (int, optional): The number of samples to use for evaluation. Defaults to 10.
        use_tqdm (bool, optional): Whether to use tqdm for progress tracking. Defaults to True.
        pbar (tqdm.tqdm, optional): An existing tqdm progress bar to use for tracking progress. Defaults to None.

    Returns:
        dict: A dictionary containing the correlations, base correlations, and normalized correlations.
    """

    correlations = [[] for _ in range(2)]
    base_correlations = []
    descs = ["", ""]

    tqdm_f = tqdm_n if is_notebook() else tqdm
    if use_tqdm:
        if pbar is None:
            pbar = tqdm_f(loader, desc="Correlation metric", total=n_samples)
            pbar_c = pbar
        else:
            pbar_c = loader
            descs[0] = pbar.desc

    for n, ((data, target), _) in enumerate(zip(pbar_c, range(n_samples))):
        data, target = data.to(device), target.to(device)
        data = process_data(data, config["data"])[0]

        base_correlations.append(get_correlation(model, data).mean(-1))
        for fixed_digit in range(2):
            f_datas = fixed_information_data(data, target, fixed_digit)
            correlations[fixed_digit].extend(
                [get_correlation(model, d).mean(-1) for d in f_datas]
            )
            descs[1] = "Correlation metric ({} / {})".format(n + 1, n_samples)
            if use_tqdm:
                pbar.set_description(descs[0] + descs[1])

    # n_timestep x n_modules x n_batch
    base_correlations = np.stack(base_correlations, -1).mean(-1)
    # n_timestep x n_modules x n_targets x n_tests
    correlations = np.stack([np.stack(corrs, -1) for corrs in correlations], 2).mean(-1)
    norm_correlations = (correlations - base_correlations[..., None]) / (
        1 - base_correlations[..., None]
    )

    return {
        "correlations": correlations,
        "base_correlations": base_correlations,
        "norm_correlations": norm_correlations,
    }
