from dynspec.models import Community
import torch

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


def get_readout_dimensions(task) : 
    if isinstance(task, list) : 
        return [get_readout_dimensions(t) for t in task]
    elif task in n_outs :
        return n_outs[task]
    else :
        raise ValueError("Unknown task: {}".format(task))


def init_model(config, device=torch.device('cpu')) : 

    readout_dim = get_readout_dimensions(config['training']["task"])
    config['readout']['output_size'] = readout_dim
    model =  Community(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config['optim'])
    return model, optimizer
