import torch
from copy import deepcopy


def get_single_task(task, target, n_classes=None):
    if n_classes is None:
        n_classes = len(target[:, 0].unique())

    # composed names of task should be with -
    digits = target.T
    try:
        t = int(task)
        return target[..., t]
    except ValueError:
        pass

    if task == "inv":
        return target.flip(-1)

    elif task in ["none", "both", "all"]:
        return target

    elif "parity" in task:
        if "digits" in task:
            parity = (digits.sum(0)) % 2  # 0 when same parity
            if "both" in task:
                return torch.stack(
                    [
                        torch.where(parity.bool(), digits[0], digits[1]),
                        torch.where(parity.bool(), digits[1], digits[0]),
                    ],
                    -1,
                )
            elif "equal" in task:
                tgt = torch.where(parity.bool(), digits[0], digits[1])
                tgt[
                    (digits[0] == digits[1])
                    | (digits[0] == (digits[1] - 1) % n_classes)
                ] = (n_classes + 1)
                return tgt
            elif "sum" in task:
                return parity
            else:
                return torch.where(parity.bool(), digits[0], digits[1])

        else:
            return target % 2

    elif task == "max":
        return target.max(-1)[0]

    elif task == "min":
        return target.min(-1)[0]

    elif task == "opposite":
        return n_classes - target - 1

    elif task == "sum":
        return target.sum(-1)

    elif task == "bitand":
        return digits[0] & digits[1]

    elif task == "bitor":
        return digits[0] | digits[1]

    elif "bitxor" in task:
        xor = digits[0] ^ digits[1]

        if "last" in task:
            n_last = int(task.split("-")[-1])
            xor = dec2bin(xor)
            xor = xor[..., -n_last:]
            xor = bin2dec(xor, n_last)

        elif "first" in task:
            n_first = int(task.split("-")[-1])
            xor = dec2bin(xor)
            xor = xor[..., :n_first]
            xor = bin2dec(xor, n_first)

        return xor

    else:
        raise ValueError("Task not recognized")


def get_task_target(target, task, n_classes):
    """
    Returns target for different possible tasks
    Args :
        targets : original digits : size (batch x 2)
        task : task to be conducted :
               digit number ("0", "1"), "parity", "parity_digits_10", "parity_digits_100" or "sum" ...
    """

    if type(task) is list:
        targets = [get_task_target(target, t, n_classes) for t in task]
        try:
            return torch.stack(targets)
        except (ValueError, RuntimeError) as e:
            return targets

    else:
        new_target = deepcopy(target)

        # Task can be a combination of subtasks, separated by _
        tasks = task.split("_")

        for task in tasks:
            new_target = get_single_task(task, new_target, n_classes)

        if len(new_target.shape) == 2:
            new_target = new_target.T

        return new_target


def dec2bin(x):
    bits = int((torch.floor(torch.log2(x)) + 1).max())
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(x.dtype)


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)
