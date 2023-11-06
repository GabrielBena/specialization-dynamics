import torch


def random_decision(outputs, p=0.5):
    """Randomly choose between two possible outputs.

    Parameters
    ----------
    outputs : torch.Tensor
        The two outputs to choose between. The first dimension should be the
        batch dimension and the second dimension should be the module dimension.
    p : float, optional
        The probability of choosing the first output, by default 0.5

    Returns
    -------
    torch.Tensor
        The chosen output.
    """
    batchs = outputs.shape[1]
    device = outputs.device
    deciding_modules = torch.rand(batchs).to(device) < p
    mask = torch.einsum(
        "ab, a -> ab", torch.ones_like(outputs[0]), deciding_modules
    ).bool()
    outputs = torch.where(mask, outputs[0, ...], outputs[1, ...])
    return outputs, deciding_modules


def max_decision(outputs):
    """
    Get the readout from module that produced the maximum overall value
    Note that this is NOT a

    Args:
        outputs (_type_): _description_

    Returns:
        _type_: _description_
    """

    device = outputs.device
    n_modules = outputs.shape[0]

    _, deciding_ags = torch.max(
        torch.stack([torch.max(outputs[i, ...], axis=-1)[0] for i in range(n_modules)]),
        axis=0,
    )
    mask_1 = deciding_ags.unsqueeze(0).unsqueeze(-1).expand_as(outputs)
    mask_2 = torch.einsum(
        "b, b... -> b...",
        torch.arange(n_modules).to(device),
        torch.ones_like(outputs),
    )
    mask = mask_1 == mask_2

    return (outputs * mask).sum(0), deciding_ags


def get_temporal_decision(outputs, temporal_decision):
    """
    Given a tensor of outputs and a temporal decision, returns a tensor of outputs based on the decision.

    Args:
        outputs (torch.Tensor): A tensor of outputs.
        temporal_decision (str or int): A string indicating the type of temporal decision to make, or an integer
            indicating the specific time step to use.

    Returns:
        torch.Tensor: A tensor of outputs based on the temporal decision.

    Raises:
        ValueError: If the temporal decision is not recognized.
    """
    try:
        deciding_ts = int(temporal_decision)
        outputs = outputs[deciding_ts]
    except (ValueError, TypeError) as e:
        if temporal_decision == "last":
            outputs = outputs[-1]
        elif temporal_decision == "sum":
            outputs = torch.sum(outputs, axis=0)
        elif temporal_decision == "mean":
            outputs = torch.mean(outputs, axis=0)
        elif temporal_decision == None or temporal_decision == "none":
            return outputs
        else:
            raise ValueError(
                'temporal decision not recognized, try "last", "sum" or "mean", or time_step of decision ("0", "-1" ) '
            )

    return outputs


def get_module_decision(outputs, module_decision):
    """
    Given a set of outputs and a module decision, returns the selected outputs and the corresponding decision indices.

    Args:
        outputs (torch.Tensor): The outputs to select from.
        module_decision (str or int or None): The decision method to use. If an integer is provided, the outputs from the
            corresponding module are selected. If "max", the outputs with the highest value are selected. If "random", a
            random output is selected. If "sum", the outputs are summed along the first dimension. If "self", the outputs
            corresponding to each agent are selected. If "both", "all", "none", or None, no outputs are selected.

    Returns:
        torch.Tensor or list of torch.Tensor or None: The selected outputs. If multiple outputs are selected, they are
            returned as a list. If no outputs are selected, None is returned.
        torch.Tensor or None: The decision indices. If multiple outputs are selected, they are returned as a list of
            indices. If no outputs are selected, None is returned.
    """
    try:
        deciding_ags = int(module_decision)
        outputs = outputs[deciding_ags]
        deciding_ags = torch.ones(outputs.shape[0]) * deciding_ags

    except ValueError:
        if module_decision == "max":
            outputs, deciding_ags = max_decision(outputs)

        elif module_decision == "random":
            outputs, deciding_ags = random_decision(outputs)

        elif module_decision == "sum":
            outputs = outputs.sum(0)
            deciding_ags = None

        elif module_decision == "self":
            assert len(outputs.shape) > 3
            outputs = [out[ag] for ag, out in enumerate(outputs)]
            deciding_ags = None
            try:
                outputs = torch.stack(outputs)
            except TypeError:
                pass

        elif module_decision in ["both", "all", "none", None]:
            deciding_ags = None

        else:
            raise ValueError(
                'Deciding module not recognized, try module number ("0", "1"), "max", "random", "both" or "sum" '
            )

    return outputs, deciding_ags


def get_decision(outputs, temporal_decision="last", module_decision="max"):
    """
    Given a set of outputs from a neural network, returns a decision based on the specified temporal and module decision methods.

    Args:
        outputs (list or tensor): The outputs from a neural network.
        temporal_decision (str): The temporal decision method to use. Can be "last" (default), "mean",  or an integer
            indicating the specific time step to use..
        module_decision (str): The module decision method to use. Can be "max" (default), "sum", or  or an integer
            indicating the specific module to use. Can also be a combination of these methods separated by underscores.

    Returns:
        tuple: A tuple containing the decision tensor and the indices of the agents that made the decision.
    """
    if isinstance(outputs, list):
        decs = [
            get_decision(out, temporal_decision, module_decision) for out in outputs
        ]
        outputs = [dec[0] for dec in decs]
        deciding_ags = [dec[1] for dec in decs]
        return torch.stack(outputs, -3), deciding_ags
    else:
        outputs = get_temporal_decision(outputs, temporal_decision)

        try:
            if len(outputs.shape) == 2:
                return outputs, None
        except AttributeError:
            pass

        for ag_decision in module_decision.split("_"):
            outputs, deciding_ags = get_module_decision(outputs, ag_decision)

        return outputs.squeeze(), deciding_ags
