import torch
import torch.nn as nn
import copy
from dynspec.training import train_community


def readout_mask(n_agents, n_in, n_out, ag_to_mask=None):
    mask = torch.ones(n_agents).unsqueeze(0)
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
    model = copy.deepcopy(original_model)
    retraining_config = config.copy()

    n_classes = config["data"]["n_classes_per_digit"]
    nb_steps = config["data"]["nb_steps"]
    n_agents = config["agents"]["n_agents"]
    n_targets = 2

    retraining_config["readout"]["n_hid"] = 30
    model.readout = nn.ModuleList(
        [
            nn.ModuleList(
                [
                    nn.Sequential(
                        *[
                            nn.Linear(
                                model.n_agents * model.hidden_size,
                                retraining_config["readout"]["n_hid"],
                            ),
                            nn.ReLU(),
                            nn.Linear(retraining_config["readout"]["n_hid"], n_classes),
                        ]
                    )
                    if retraining_config["readout"]["n_hid"]
                    else nn.Linear(model.n_agents * model.hidden_size, n_classes)
                    for _ in range(3)
                ]
            )
            for _ in range(n_targets)
        ]
    )

    for n, p in model.named_parameters():
        if not "readout" in n:
            p.requires_grad = False
        else:
            p.requires_grad = True
    model.register_buffer(
        "output_mask",
        torch.stack(
            [
                torch.stack(
                    [
                        readout_mask(
                            model.n_agents,
                            model.hidden_size,
                            retraining_config["readout"]["n_hid"],
                            ag_to_mask=i,
                        )
                        if retraining_config["readout"]["n_hid"] is not None
                        else readout_mask(
                            model.n_agents,
                            model.hidden_size,
                            n_classes,
                            ag_to_mask=i,
                        )
                        for i in list(range(model.n_agents))[::-1] + [None]
                    ]
                )
                for _ in range(n_targets)
            ]
        ),
    )

    model.to(device)

    retraining_config["decision"] = ("none", "none")

    retraining_config["training"]["task"] = [
        [[str(i) for i in range(n_targets)] for _ in range(n_agents + 1)]
        for _ in range(nb_steps)
    ]
    model.output_size = [
        [10 for i in list(range(model.n_agents))[::-1] + [None]]
        for _ in range(n_targets)
    ]
    retraining_config["training"]["n_epochs"] = 5

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
