import torch
from copy import deepcopy


def get_single_task(task, target, n_classes=None):
    """
    Returns a single task based on the given task name and target.

    Args:
        task (str): Name of the task to perform.
        target (torch.Tensor): Target tensor.
        n_classes (int, optional): Number of classes. Defaults to None.

    Returns:
        torch.Tensor: Single task tensor.

    Possible tasks:
        - "inv" : flip target digits
        - "none, "both", "all" : do nothing
        - "parity-digits" : flip digits based on parity
        - "parity-digits-both" : double target : flip digits based on parity and 1-parity
        - "parity-digits-sum" : return parity of sum of digits
        - "max" : return max of digits
        - "min" : return min of digits
        - "opposite" : return n_classes - target - 1
        - "sum" : return sum of digits
        - "bitand" : return bitwise and of digits
        - "bitor" : return bitwise or of digits
        - "bitxor-last-N" : return bitwise xor of digits, keeping only last N bits
        - "bitxor-first-N" : return bitwise xor of digits, keeping only first N bits
    """
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
                return [
                    torch.where(parity.bool(), digits[0], digits[1]),
                    torch.where(parity.bool(), digits[1], digits[0]),
                ]
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
    Returns the target tensor for a given task.

    Args:
        target (torch.Tensor): The original target tensor.
        task (str or list): The task(s) to perform on the target tensor.
        n_classes (int): The number of classes in the target tensor.

    Returns:
        torch.Tensor: The target tensor after performing the specified task(s).
    """

    if type(task) is list:
        targets = [get_task_target(target, t, n_classes) for t in task]
        # try:
        #     return torch.stack(targets)
        # except (ValueError, RuntimeError) as e:
        #     return targets
        return targets

    else:
        new_target = deepcopy(target)

        # Task can be a combination of subtasks, separated by _
        tasks = task.split("_")

        for task in tasks:
            new_target = get_single_task(task, new_target, n_classes)

        if hasattr(new_target, "shape") and len(new_target.shape) == 2:
            new_target = new_target.T
            # new_target = list(new_target.split(1, 0))

        return new_target


def dec2bin(x):
    """
    Converts a decimal number to its binary representation.

    Args:
        x (torch.Tensor): The decimal number to convert.

    Returns:
        torch.Tensor: The binary representation of the input decimal number.
    """
    bits = int((torch.floor(torch.log2(x)) + 1).max())
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(x.dtype)


def bin2dec(b, bits):
    """
    Converts a binary tensor to a decimal tensor.

    Args:
        b (torch.Tensor): The binary tensor to convert.
        bits (int): The number of bits in the binary tensor.

    Returns:
        torch.Tensor: The decimal tensor.
    """
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)
