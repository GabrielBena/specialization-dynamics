import torch


def random_decision(outputs, p=0.5):
    """Randomly choose between two possible outputs.

    Parameters
    ----------
    outputs : torch.Tensor
        The two outputs to choose between. The first dimension should be the
        batch dimension and the second dimension should be the agent dimension.
    p : float, optional
        The probability of choosing the first output, by default 0.5

    Returns
    -------
    torch.Tensor
        The chosen output.
    """
    batchs = outputs.shape[1]
    device = outputs.device
    deciding_agents = torch.rand(batchs).to(device) < p
    mask = torch.einsum(
        "ab, a -> ab", torch.ones_like(outputs[0]), deciding_agents
    ).bool()
    outputs = torch.where(mask, outputs[0, ...], outputs[1, ...])
    return outputs, deciding_agents


def max_decision(outputs):
    device = outputs.device
    n_agents = outputs.shape[0]

    _, deciding_ags = torch.max(
        torch.stack([torch.max(outputs[i, ...], axis=-1)[0] for i in range(n_agents)]),
        axis=0,
    )
    mask_1 = deciding_ags.unsqueeze(0).unsqueeze(-1).expand_as(outputs)
    mask_2 = torch.einsum(
        "b, b... -> b...",
        torch.arange(n_agents).to(device),
        torch.ones_like(outputs),
    )
    mask = mask_1 == mask_2

    return (outputs * mask).sum(0), deciding_ags


def get_temporal_decision(outputs, temporal_decision):
    try:
        deciding_ts = int(temporal_decision)
        outputs = outputs[deciding_ts]
    except ValueError:
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


def get_agent_decision(outputs, agent_decision, target=None):
    try:
        deciding_ags = int(agent_decision)
        outputs = outputs[deciding_ags]
        deciding_ags = torch.ones(outputs.shape[0]) * deciding_ags

    except ValueError:
        if agent_decision == "max":
            outputs, deciding_ags = max_decision(outputs)

        elif agent_decision == "random":
            outputs, deciding_ags = random_decision(outputs)

        elif agent_decision == "sum":
            outputs = outputs.sum(0)
            deciding_ags = None

        elif agent_decision == "self":
            assert len(outputs.shape) > 3
            outputs = [out[ag] for ag, out in enumerate(outputs)]
            deciding_ags = None
            try:
                outputs = torch.stack(outputs)
            except TypeError:
                pass

        elif agent_decision in ["both", "all", "none", None]:
            deciding_ags = None

        else:
            raise ValueError(
                'Deciding agent not recognized, try agent number ("0", "1"), "max", "random", "both" or "sum" '
            )

    return outputs, deciding_ags


def get_decision(outputs, temporal_decision="last", agent_decision="0"):
    outputs = get_temporal_decision(outputs, temporal_decision)

    try:
        if len(outputs.shape) == 2:
            return outputs, None
    except AttributeError:
        pass

    for ag_decision in agent_decision.split("_"):
        outputs, deciding_ags = get_agent_decision(outputs, ag_decision)

    return outputs, deciding_ags
