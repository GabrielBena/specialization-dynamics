import torch
import torch.nn as nn
import numpy as np
from itertools import product
from torch.nn.utils.parametrize import register_parametrization as rpm
from dynspec.surrogate import super_spike
from quant.quantization import Quantization

cell_types_dict = {str(t): t for t in [nn.RNN, nn.GRU, nn.RNNCell, nn.GRUCell]}


def state_mask(n_modules, n_0, n_1, gru=False):
    """
    Create the mask for the state-state connections (iter-layers / intra-modules), making sure modules only connect to themselves

    Args:
        n_modules (int): number of modules
        n_0 (int): number of input features
        n_1 (int): number of output features
        gru (bool, optional): model is GRU. Defaults to False.

    Returns:
        mask (torch.tensor): mask for the inter-layers / intra-modules connections
    """
    mask = torch.eye(n_modules)
    mask = mask.repeat_interleave(n_0, 0).repeat_interleave(n_1, 1)
    if gru:
        mask = torch.concat([m for m in mask.unsqueeze(0).repeat_interleave(3, 0)])
    return mask


def sparse_mask(sparsity, n_in, n_out):
    """
    Create the sparse masks for inter-module connections

    Args:
        sparsity (float): sparsity of connection, 0 = no connection, 1 = fully connected
        n_in (int): number of input features
        n_out (int): number of output features

    Returns:
        w_mask (torch.tensor): mask for the connections
    """
    if sparsity >= 0:
        assert sparsity <= 1
        nb_non_zero = int(sparsity * n_in * n_out)
    else:
        nb_non_zero = -sparsity
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


def comms_mask(sparsity, n_modules, n_hidden, gru=False):
    """
    Create the masks for both inter-module and intra-module connections

    Args:
        sparsity (float): sparsity of connection inter-modules, 0 = no connection, 1 = fully connected
        n_modules (int): number of modules
        n_hidden (int): number of hidden features
        gru (bool, optional): use GRU cell. Defaults to False.

    Returns:
        masks (list): list of masks for the inter-module and intra-module connections
    """
    comms_mask = torch.zeros((n_modules * n_hidden, n_modules * n_hidden))
    rec_mask = torch.zeros((n_modules * n_hidden, n_modules * n_hidden))

    for i, j in product(range(n_modules), repeat=2):
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


class Masked_weight(nn.Module):
    """
    Parametrization of the weights of a layer with a mask

    Args:
        mask (torch.tensor): mask for the weights
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, W):
        W = W * self.mask
        return W


class Quantized_weight(nn.Module):
    """
    Parametrization of the weights of a layer with a quantization

    Args: n_bits (int): number of bits for the quantization
    """

    def __init__(self, n_bits):
        super().__init__()
        self.quant_values = np.linspace(
            -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1, 2**n_bits
        )
        self.quant = Quantization(
            quant_values=self.quant_values,
        )

    def forward(self, W):
        return self.quant(W)


class Scaled_mask_weight(nn.Module):
    """
    Parametrization of the weights of a layer with a mask and a scale

    Args:
        mask (torch.tensor): mask for the weights
        scale (float): scale of the weights
    """

    def __init__(self, mask, scale) -> None:
        super().__init__()
        scaled_mask = torch.ones_like(mask) + mask * (scale - 1)
        self.register_buffer("scaled_mask", scaled_mask)

    def forward(self, W):
        return W * self.scaled_mask


def reccursive_rpm(model, masks):
    """
    Apply reccursive weight parametrization for a (potentially nested) list of modules

    Args:
        model (nn.Module or nn.ModuleList or nn.Sequential): model to parametrize
        masks (list or torch.tensor): mask(s) to use for parametrization

    Returns:
        model: parametrized model
    """
    if isinstance(model, nn.ModuleList):
        [reccursive_rpm(m, mask) for m, mask in zip(model, masks)]
    elif isinstance(model, nn.Sequential):
        if isinstance(masks, (list, tuple)):
            assert len([m for m in model if hasattr(m, "weight")]) == len(masks)
            [
                reccursive_rpm(m, mask)
                for m, mask in zip([m for m in model if hasattr(m, "weight")], masks)
            ]
        else:
            rpm(model[0], "weight", Masked_weight(masks))
    else:
        rpm(model, "weight", Masked_weight(masks))
    return model


class Readout(nn.Module):
    """
    Readout module, comprising potentially nested list of linear layers
    """

    def __init__(
        self, readout_config, n_modules, ag_hidden_size, out_masks=None
    ) -> None:
        super().__init__()

        self.n_hid = readout_config.get("n_hid", None)
        self.common_readout = readout_config.get("common_readout", False)
        self.output_size = readout_config["output_size"]
        self.n_modules = n_modules
        self.ag_hidden_size = ag_hidden_size
        self.retraining = readout_config.get("retraining", False)

        self.out_masks = (
            self.create_mask(self.output_size, self.n_hid)
            if out_masks is None
            else out_masks
        )
        self.layers = self.create_readout(self.output_size, self.n_hid)
        reccursive_rpm(self.layers, self.out_masks)

    def create_mask(self, output_size, n_hid=None):
        if not isinstance(output_size, (tuple, list)):
            if n_hid is not None:
                mask = (
                    state_mask(self.n_modules, n_hid, self.ag_hidden_size),
                    state_mask(self.n_modules, output_size, n_hid),
                )
            else:
                mask = state_mask(self.n_modules, output_size, self.ag_hidden_size)
            return mask
        else:
            if n_hid is not None:
                return [self.create_mask(o, n_h) for o, n_h in zip(output_size, n_hid)]
            else:
                return [self.create_mask(o) for o in output_size]

    def create_readout(self, output_size, n_hid=None):
        if not isinstance(output_size, (tuple, list)):
            if n_hid is not None:
                return nn.Sequential(
                    nn.Linear(
                        self.n_modules * self.ag_hidden_size,
                        n_hid * self.n_modules if not self.retraining else n_hid,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        n_hid * self.n_modules if not self.retraining else n_hid,
                        (
                            output_size * self.n_modules
                            if not self.retraining
                            else output_size
                        ),
                    ),
                )
            else:
                return nn.Linear(
                    self.n_modules * self.ag_hidden_size,
                    (
                        output_size * self.n_modules
                        if not self.retraining
                        else output_size
                    ),
                )
        else:
            if n_hid is not None:
                return nn.ModuleList(
                    [self.create_readout(o, n_h) for o, n_h in zip(output_size, n_hid)]
                )
            else:
                return nn.ModuleList([self.create_readout(o) for o in output_size])

    def process_readout(self, input, readout, output_size):
        """
        Single readout processing

        Args:
            input (torch.tensor): input to the readout
            readout (nn.Module): readout module
            common_readout (bool): whether the readout is common to all modules
            output_size (int): size of the output

        Returns:
            output (torch.tensor): processed output
        """
        output = readout(input)
        if not self.common_readout:
            output = torch.stack(output.split(output_size, -1), 1)
        return output

    def reccursive_readout(self, input, readout, output_size):
        """
        Compute readout reccursively, for a (potentially nested) list of readouts

        Args:
            input (torch.tensor): input to the readout
            readout (nn.Module or nn.ModuleList): readout module(s)
            common_readout (bool): whether the readout is common to all modules
            output_size (int): size of the output

        Returns:
            out (torch.tensor): processed output
        """
        if isinstance(readout, nn.ModuleList):
            out = [
                self.reccursive_readout(input, r, size)
                for r, size in zip(readout, output_size)
            ]
        else:
            out = self.process_readout(input, readout, output_size)

        return out

    def forward(self, input):
        return self.reccursive_readout(input, self.layers, self.output_size)


class BinaryComms(nn.Module):
    """
    Binary communication module
    """

    def __init__(self, comms) -> None:
        super().__init__()
        self.comms = comms

    def forward(self, input):
        out = super_spike(self.comms(input)[0])
        return out


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.weights_ih = nn.ParameterList(
            [
                nn.Parameter(
                    torch.Tensor(input_size if i == 0 else hidden_size, hidden_size)
                )
                for i in range(num_layers)
            ]
        )
        self.weights_hh = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        self.bias_ih = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for _ in range(num_layers)]
        )
        self.bias_hh = nn.ParameterList(
            [nn.Parameter(torch.Tensor(hidden_size)) for _ in range(num_layers)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        for i in range(self.num_layers):
            h[i] = torch.tanh(
                torch.mm(x, self.weights_ih[i])
                + self.bias_ih[i]
                + torch.mm(h[i - 1], self.weights_hh[i])
                + self.bias_hh[i]
            )
            h[i] = self.dropout(h[i])
        return h[-1]


class Community(nn.Module):
    """
    A PyTorch module representing a community of neural network modules.

    Args:
        config (dict): A dictionary containing the configuration parameters for the community.
            The dictionary should have the following keys:
                - "input": A dictionary containing the input configuration parameters. It should have the following keys:
                    - "input_size" (int): The size of the input.
                    - "common_input" (bool): Whether the input is shared across all modules.
                - "modules": A dictionary containing the module configuration parameters. It should have the following keys:
                    - "n_modules" (int): The number of modules in the community.
                    - "hidden_size" (int): The number of features in the hidden state of each module.
                    - "n_layers" (int): The number of layers in each module.
                    - "dropout" (float): The dropout probability.
                    - "cell_type" (str): The type of reccurent cell to use (RNN or GRU).
                - "connections": A dictionary containing the connection configuration parameters. It should have the following keys:
                    - "sparsity" (float): The sparsity of the connection matrix.
                - "readout": A dictionary containing the readout configuration parameters. It should have the following keys:
                    - "output_size" (int or list of ints): The size of the output(s).
                    - "common_readout" (bool): Whether the readout is shared across all modules.
                    - "n_hid" (int or None): The number of hidden units in the readout layer(s).

    Attributes:
        input_size (int): The size of the input.
        common_input (bool): Whether the input is shared across all modules.
        n_modules (int): The number of modules in the community.
        hidden_size (int): The number of features in the hidden state of each module.
        n_layers (int): The number of layers in each module.
        dropout (float): The dropout probability.
        cell_type (str): The type of RNN cell to use.
        sparsity (float): The sparsity of the connection matrix.
        output_size (int or list of ints): The size of the output(s).
        common_readout (bool): Whether the readout is shared across all modules.
        masks (dict): A dictionary containing the masks used for computing the forward pass.
        core (nn.Module): The core RNN module.
        readout (Readout): The readout module.
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        (
            self.input_config,
            self.modules_config,
            self.connections_config,
            self.readout_config,
        ) = [config[k] for k in ["input", "modules", "connections", "readout"]]

        self.input_size, self.common_input = [
            self.input_config[k] for k in ["input_size", "common_input"]
        ]
        (
            self.n_modules,
            self.hidden_size,
            self.n_layers,
            self.dropout,
            self.cell_type,
        ) = [
            self.modules_config[k]
            for k in ["n_modules", "hidden_size", "n_layers", "dropout", "cell_type"]
        ]
        self.sparsity, self.binary_comms = self.connections_config[
            "sparsity"
        ], self.connections_config.get("binary", False)

        self.output_size, self.common_readout = [
            self.readout_config[k] for k in ["output_size", "common_readout"]
        ]

        gru = "GRU" in self.cell_type
        rec_masks = comms_mask(self.sparsity, self.n_modules, self.hidden_size, gru=gru)

        out_size = (
            self.output_size
            if self.readout_config.get("n_hid", None) is None
            else self.readout_config["n_hid"]
        )
        if (
            isinstance(self.output_size, list)
            and self.readout_config.get("n_hid", None) is not None
        ):
            out_size = [out_size for _ in self.output_size]

        self.masks = {
            "input_mask": (
                state_mask(self.n_modules, self.hidden_size, self.input_size, gru=gru)
                if not self.common_input
                else torch.ones_like(
                    state_mask(
                        self.n_modules, self.hidden_size, self.input_size, gru=gru
                    )
                )
            ),
            "state_mask": state_mask(
                self.n_modules, self.hidden_size, self.hidden_size, gru=gru
            ),
            "rec_mask": rec_masks[1],
            "comms_mask": rec_masks[0],
        }

        self.core = cell_types_dict[self.cell_type](
            input_size=self.input_size * self.n_modules,
            hidden_size=self.hidden_size * self.n_modules,
            num_layers=self.n_layers,
            batch_first=False,
            bias=False,
            dropout=self.dropout,
        )

        self.comms = cell_types_dict[self.cell_type](
            input_size=self.input_size * self.n_modules,
            hidden_size=self.hidden_size * self.n_modules,
            num_layers=self.n_layers,
            batch_first=False,
            bias=False,
            dropout=self.dropout,
        )

        for n, p in self.core.named_parameters():
            getattr(self.comms, n).data = p.data

        self.readout = Readout(self.readout_config, self.n_modules, self.hidden_size)
        for n, m in self.masks.items():
            self.register_buffer(n, m)

        for n in dict(self.core.named_parameters()).copy().keys():
            if "weight_hh" in n:
                rpm(self.core, n, Masked_weight(self.rec_mask))
            #     if n[-1] == str(self.n_layers - 1):
            #         rpm(self.core, n, Masked_weight(self.comms_mask + self.rec_mask))
            #     else:
            #         rpm(self.core, n, Masked_weight(self.rec_mask))
            elif "weight_ih" in n and n[-1] == "0":
                rpm(self.core, n, Masked_weight(self.input_mask))
            elif "weight_ih" in n and n[-1] != "0":
                rpm(self.core, n, Masked_weight(self.state_mask))

        for n in dict(self.comms.named_parameters()).copy().keys():
            if "weight_hh" in n:
                if n[-1] == str(self.n_layers - 1):
                    rpm(self.comms, n, Masked_weight(self.comms_mask))
                else:
                    rpm(self.comms, n, Masked_weight(torch.zeros_like(self.comms_mask)))
            elif "weight_ih" in n and n[-1] == "0":
                rpm(self.comms, n, Masked_weight(self.input_mask))
            elif "weight_ih" in n and n[-1] != "0":
                rpm(self.comms, n, Masked_weight(self.state_mask))

        if self.binary_comms:
            self.comms = BinaryComms(self.comms)

    @property
    def multi_readout(self):
        if hasattr(self, "readout"):
            return isinstance(self.readout, nn.ModuleList)
        else:
            return isinstance(self.output_size, list)

    def forward(self, input):
        if "Cell" in self.cell_type:
            all_states, states = [], None
            for t, t_input in enumerate(input):
                states = self.core(t_input, states) + self.comms(t_input, states)
                all_states.append(states)
            all_states = torch.stack(all_states)
        else:
            core_out, comms_out = self.core(input), self.comms(input)
            all_states, final_states = (
                core_out[0] + comms_out[0],
                core_out[1] + comms_out[1],
            )

        outputs = self.readout(all_states)
        return outputs, all_states


def init_model(config, device=torch.device("cpu")):
    """
    Initializes a model and optimizer based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration for the model.
        device (torch.device): The device to use for the model (default is CPU).

    Returns:
        tuple: A tuple containing the initialized model and optimizer.
    """
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
    if config["training"]["task"] == "parity-digits-both":
        config["training"]["task"] = ["parity-digits", "inv-parity-digits"]

    def get_readout_dimensions(task, n_hid=None):
        """
        Get the output dimensions for a given task.
        """
        if isinstance(task, list):
            out_dims = [get_readout_dimensions(t, n_hid) for t in task]
        elif task in n_outs:
            out_dims = n_outs[task], n_hid
        else:
            raise ValueError("Unknown task: {}".format(task))

        if isinstance(out_dims, tuple):
            return out_dims
        else:
            return tuple(zip(*out_dims))

    readout_dim, readout_n_hidden = get_readout_dimensions(
        config["training"]["task"], config["readout"].get("n_hid", None)
    )
    config["readout"]["output_size"] = readout_dim
    config["readout"]["n_hid"] = readout_n_hidden
    model = Community(config).to(device)
    gamma = config["optim"].pop("gamma", None)
    optimizer = torch.optim.AdamW(model.parameters(), **config["optim"])
    if gamma:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    else:
        scheduler = None
    return model, optimizer, scheduler
