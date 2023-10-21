import torch
import torch.nn as nn
import numpy as np
from itertools import product


def state_mask(n_agents, n_0, n_1, gru=False):
    # Mask for the state to state connections between layers
    mask = torch.eye(n_agents)
    mask = mask.repeat_interleave(n_0, 0).repeat_interleave(n_1, 1)
    if gru:
        mask = torch.concat([m for m in mask.unsqueeze(0).repeat_interleave(3, 0)])
    return mask


def sparse_mask(sparsity, n_in, n_out):
    nb_non_zero = int(sparsity * n_in * n_out)
    w_mask = np.zeros((n_in, n_out), dtype=bool)
    # ind_in = rd.choice(np.arange(in_features),size=self.nb_non_zero)
    # ind_out = rd.choice(np.arange(out_features),size=self.nb_non_zero)

    ind_in, ind_out = np.unravel_index(
        np.random.choice(np.arange(n_in * n_out), nb_non_zero, replace=False),
        (n_in, n_out),
    )
    w_mask[ind_in, ind_out] = True
    w_mask = torch.tensor(w_mask)

    return w_mask


def comms_mask(sparsity, n_agents, n_hidden, gru=False):
    comms_mask = torch.zeros((n_agents * n_hidden, n_agents * n_hidden))
    rec_mask = torch.zeros((n_agents * n_hidden, n_agents * n_hidden))

    for i, j in product(range(n_agents), repeat=2):
        if i != j:
            comms_mask[
                i * n_hidden : (i + 1) * n_hidden, j * n_hidden : (j + 1) * n_hidden
            ] = sparse_mask(sparsity, n_hidden, n_hidden)
        else:
            rec_mask[
                i * n_hidden : (i + 1) * n_hidden, j * n_hidden : (j + 1) * n_hidden
            ] = 1 - torch.eye(n_hidden)

    masks = [comms_mask, rec_mask]
    if gru:
        masks = [
            torch.concat([m for m in mask.unsqueeze(0).repeat_interleave(3, 0)])
            for mask in masks
        ]

    return masks


class masked_RNN(nn.RNN):
    def __init__(self, *args, masks, **kwargs):
        super().__init__(*args, **kwargs)
        [self.register_buffer(n, m) for n, m in masks.items()]

    def forward(self, input, hx=None):
        for n, p in self.named_parameters():
            if "weight_hh" in n:
                if n[-1] == str(self.num_layers - 1):
                    p.data *= self.comms_mask + self.rec_mask
                else:
                    p.data *= self.rec_mask
            elif "weight_ih" in n and n[-1] != "0":
                # print(p.shape, self.input_mask.shape)
                p.data *= self.state_mask
            elif "weight_ih" in n and n[-1] == "0":
                # print(p.shape, self.input_mask.shape)
                p.data *= self.input_mask

        return super().forward(input, hx=hx)


class masked_GRU(nn.GRU):
    def __init__(self, *args, masks, **kwargs):
        super().__init__(*args, **kwargs)
        [self.register_buffer(n, m) for n, m in masks.items()]

    def forward(self, input, hx=None):
        for n, p in self.named_parameters():
            if "weight_hh" in n:
                if n[-1] == str(self.num_layers - 1):
                    p.data *= self.comms_mask + self.rec_mask
                else:
                    p.data *= self.rec_mask
            elif "weight_ih" in n and n[-1] != "0":
                # print(p.shape, self.input_mask.shape)
                p.data *= self.state_mask
            elif "weight_ih" in n and n[-1] == "0":
                # print(p.shape, self.input_mask.shape)
                p.data *= self.input_mask

        return super().forward(input, hx=hx)


cell_types_dict = {str(t): t for t in [masked_RNN, masked_GRU]}


def reccursive_readout(input, readout, common_readout, output_size):
    if isinstance(readout, nn.ModuleList):
        out = [
            reccursive_readout(input, r, common_readout, size)
            for r, size in zip(readout, output_size)
        ]
    else:
        out = process_readout(input, readout, common_readout, output_size)

    return out


def process_readout(input, readout, common_readout, output_size):
    output = readout(input)
    if not common_readout:
        output = torch.stack(output.split(output_size, -1), 1)
    return output


def reccursive_masking(model, masks):
    if isinstance(model, nn.ModuleList):
        [reccursive_masking(m, mask) for m, mask in zip(model, masks)]
    elif isinstance(model, nn.Sequential):
        model[0].weight.data *= masks
    else:
        model.weight.data *= masks
    return model


class Community(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        (
            self.input_config,
            self.agents_configs,
            self.connections_config,
            self.readout_config,
        ) = [config[k] for k in ["input", "agents", "connections", "readout"]]

        self.is_community = True
        self.input_size, self.common_input = [
            self.input_config[k] for k in ["input_size", "common_input"]
        ]
        self.n_agents, self.hidden_size, self.n_layers, self.dropout, self.cell_type = [
            self.agents_configs[k]
            for k in ["n_agents", "hidden_size", "n_layers", "dropout", "cell_type"]
        ]
        self.sparsity = self.connections_config["sparsity"]
        self.output_size, self.common_readout = [
            self.readout_config[k] for k in ["output_size", "common_readout"]
        ]

        gru = "GRU" in self.cell_type
        rec_masks = comms_mask(self.sparsity, self.n_agents, self.hidden_size, gru=gru)

        self.masks = {
            "input_mask": state_mask(
                self.n_agents, self.hidden_size, self.input_size, gru=gru
            )
            if not self.common_input
            else torch.ones_like(
                state_mask(self.n_agents, self.hidden_size, self.input_size, gru=gru)
            ),
            "state_mask": state_mask(
                self.n_agents, self.hidden_size, self.hidden_size, gru=gru
            ),
            "rec_mask": rec_masks[1],
            "comms_mask": rec_masks[0],
            "output_mask": state_mask(self.n_agents, self.output_size, self.hidden_size)
            if not isinstance(self.output_size, list)
            else torch.stack(
                [
                    state_mask(self.n_agents, o, self.hidden_size)
                    for o in self.output_size
                ]
            ),
        }

        self.core = cell_types_dict[self.cell_type](
            input_size=self.input_size * self.n_agents,
            hidden_size=self.hidden_size * self.n_agents,
            num_layers=self.n_layers,
            batch_first=False,
            bias=False,
            masks=self.masks,
            dropout=self.dropout,
        )

        for n, m in self.masks.items():
            self.register_buffer(n, m)

        if self.common_readout:
            self.readout = (
                nn.Linear(self.n_agents * self.hidden_size, self.output_size)
                if not self.multi_readout
                else nn.ModuleList(
                    [
                        nn.Linear(self.n_agents * self.hidden_size, o)
                        for o in self.output_size
                    ]
                )
            )
        else:
            self.readout = (
                nn.Linear(
                    self.n_agents * self.hidden_size, self.output_size * self.n_agents
                )
                if not self.multi_readout
                else nn.ModuleList(
                    [
                        nn.Linear(self.n_agents * self.hidden_size, o * self.n_agents)
                        for o in self.output_size
                    ]
                )
            )

    @property
    def multi_readout(self):
        if hasattr(self, "readout"):
            return isinstance(self.readout, nn.ModuleList)
        else:
            return isinstance(self.output_size, list)

    def forward(self, input):
        output, states = self.core(input)

        if not self.common_readout:
            reccursive_masking(self.readout, self.output_mask)

        output = reccursive_readout(
            output,
            self.readout,
            self.common_readout,
            self.output_size,
        )
        return output, states
