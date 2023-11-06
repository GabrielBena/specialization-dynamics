import torch
import numpy as np
from scipy.stats import pearsonr
from tqdm.notebook import tqdm as tqdm_n
from tqdm.notebook import tqdm
from dynspec.data_process import process_data
from dynspec.training import is_notebook


def fixed_information_data(
    data, target, fixed, fixed_mode="label", permute_other=True, n_agents=2
):
    data = data.clone()
    # Return a modified version of data sample, where one quality is fixed (digit label, or parity, etc)
    digits = torch.split(target, 1, dim=-1)
    bs = digits[0].shape[0]

    classes = [d.unique() for d in digits]

    if len(data.shape) == 3:
        # data = torch.stack([data for _ in range(n_agents)], 1)
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

    """
    if 0 in [d.shape[2] for d in new_data]:
        return fixed_information_data(
            data, target, fixed, fixed_mode, permute_other, n_agents
        )
    """

    return new_data


v_pearsonr = np.vectorize(pearsonr, signature="(n1),(n2)->(),()")


def randperm_no_fixed(n):
    perm = torch.randperm(n)

    if (torch.arange(n) == perm).any() and n > 4:
        return randperm_no_fixed(n)
    else:
        return perm


def get_correlation(model, data):
    states = model(data)[1]

    agent_states = states.split(model.hidden_size, -1)
    agent_states = [ag_s.cpu().data.numpy() for ag_s in agent_states]
    perm = randperm_no_fixed(agent_states[0].shape[1])
    corr = np.stack([v_pearsonr(ag_s, ag_s[:, perm])[0] for ag_s in agent_states], 1)
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

    # n_timestep x n_agents x n_batch
    base_correlations = np.stack(base_correlations, -1).mean(-1)
    # n_timestep x n_agents x n_targets x n_tests
    correlations = np.stack([np.stack(corrs, -1) for corrs in correlations], 2).mean(-1)
    norm_correlations = (correlations - base_correlations[..., None]) / (
        1 - base_correlations[..., None]
    )

    return {
        "correlations": correlations,
        "base_correlations": base_correlations,
        "norm_correlations": norm_correlations,
    }
