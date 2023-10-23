import torch
import numpy as np

""" --------------------------------------------------------------------
Dataset process utility functions
""" 


def flatten_list(l):
    """
    Flattens reccursively a list of lists
    """
    assert isinstance(l, list)

    if isinstance(l[0], list):
        return flatten_list([item for sublist in l for item in sublist])
    else:
        return l


def add_structured_noise(data, n_samples=5, noise_ratio=0.9):
    """
    Generate a noisy version of a batch of samples, by adding noise from other samples in the batch

    Args:
        data (torch.Tensor): input (noiseless) data
        n_samples (int, optional): number of other samples from batch to use as noise Defaults to 5.
        noise_ratio (float, optional): ratio of noise to add to original data. Defaults to 0.9.

    Returns:
        torch.Tensor: noisy version of the input batch
    """
    noised_idxs = np.stack(
        [
            np.random.choice(data.shape[0], size=n_samples, replace=False)
            for _ in range(data.shape[0])
        ]
    )
    noised_samples = data[noised_idxs] * (
        torch.rand([n_samples] + list(data.shape[1:])).to(data.device) < (1 / n_samples)
    )
    noised_data = (1 - noise_ratio) * data + noise_ratio * noised_samples.mean(1)
    return noised_data, noised_idxs, noised_samples


def add_temporal_noise(
    data, n_samples=5, noise_ratio=0.9, random_start=False, common_input=False
):
    """
    Create a noisy version of a temporal input sequence, by adding random noise at every step.

    Args:
        data (torch.Tensor): input batch
        random_start (bool, optional): Apply random timings to turn data ON from pure noise background. Defaults to False.
        common_input (bool, optional): Wether input data is shared across modules or separate. Defaults to False.

    Returns:
        tuple: temporal version of the noisy data, and possible random timings
    """
    # data should be shape n_steps x (n_agents) x n_sample x n_features

    data = data.transpose(1, 2)  # n_steps x n_samples x (n_agents) x n_features
    noise_data = torch.stack(
        [add_structured_noise(d, n_samples, noise_ratio)[0] for d in data]
    ).transpose(1, 2)
    nb_steps = data.shape[0]

    if random_start:
        if common_input:
            start_times = torch.randint(1, data.shape[0] - 1, (data.shape[1],))
            mask = (
                torch.arange(nb_steps)[:, None]
                >= start_times[
                    None,
                    :,
                ]
            )
        else:
            start_times = torch.randint(
                1, data.shape[0] - 1, (data.shape[1], data.shape[2])
            )
            mask = (
                torch.arange(nb_steps)[:, None, None]
                >= start_times[
                    None,
                    :,
                ]
            )

        mask = mask[..., None].transpose(1, 2).to(data.device)
        pure_noise = torch.stack(
            [add_structured_noise(d, n_samples, 1.0)[0] for d in data]
        ).transpose(1, 2)
        noise_data = mask * noise_data + (~mask) * pure_noise

        return noise_data, start_times
    else:
        return noise_data, None


def temporal_data(data, nb_steps=2, flatten=True, noise_ratio=None, random_start=False):
    """
    Stack input data in time for use with RNNs

    Args:
        data (torch.tensor): input (non-temporal) batch
        nb_steps (int, optional): number of computation steps. Defaults to 2.
        flatten (bool, optional): Defaults to True.
        noise_ratio (_type_, optional): adds noise at every time-step. Defaults to None.
        random_start (bool, optional): add stochastic dynamics, where inputs are turned ON at random times from pure noise. Defaults to False.

    Returns:
        tuple: temporal version of the data, and possible random ON timings
    """
    is_list = type(data) is list
    if flatten and not is_list:
        if data.shape[1] == 1:
            data = data.flatten(start_dim=1)
        else:
            data = data.flatten(start_dim=2)

    data = [data for _ in range(nb_steps)]
    if not is_list:
        data = torch.stack(data)
    if not is_list and len(data.shape) > 3 and flatten:
        data = data.transpose(1, 2)

    if noise_ratio is not None:
        data, start_times = add_temporal_noise(
            data, n_samples=5, noise_ratio=noise_ratio, random_start=random_start
        )
    else:
        start_times = None

    return data, start_times


def process_data(data, data_config):
    """_summary_

    Args:
        data (_type_): _description_
        data_config (_type_): _description_

    Returns:
        _type_: _description_
    """
    start_times = None
    nb_steps, noise_ratio, random_start = (
        data_config["nb_steps"],
        data_config["noise_ratio"],
        data_config["random_start"],
    )
    data, start_times = temporal_data(
        data,
        nb_steps=nb_steps,
        noise_ratio=noise_ratio,
        random_start=random_start,
    )

    data = data.transpose(1, 2)
    data = data.reshape(data.shape[0], data.shape[1], -1)

    return data, start_times
