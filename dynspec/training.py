import torch
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm as tqdm_n
from tqdm import tqdm
import copy

from dynspec.data_process import process_data
from dynspec.decision import get_decision
from dynspec.tasks import get_task_target


def is_notebook():
    try:
        get_ipython()
        notebook = True
    except NameError:
        notebook = False
    return notebook


def nested_round(acc):
    try:
        round = np.round(np.array(acc) * 100, 0).astype(float)
        if isinstance(round, np.ndarray):
            round = round.tolist()
        return round
    except TypeError:
        return [nested_round(a) for a in acc]


def nested_mean(losses):
    try:
        return torch.mean(losses)
    except TypeError:
        return torch.stack([nested_mean(l) for l in losses]).mean()


def get_loss(output, t_target, use_both=False):
    if use_both:
        loss = [get_loss(o, t_target) for o in output]
    else:
        try:
            loss = F.cross_entropy(output, t_target, reduction="none")
            output = output.unsqueeze(0)

        except (TypeError, RuntimeError) as _:
            loss = [get_loss(o, t) for o, t in zip(output, t_target)]

    try:
        loss = torch.stack(loss)
    except (RuntimeError, TypeError) as _:
        pass

    return loss


def get_acc(output, t_target, use_both=False):
    if use_both:
        acc = [get_acc(o, t_target) for o in output]
    else:
        try:
            pred = output.argmax(
                dim=-1, keepdim=True
            )  # get the index of the max log-probability

            correct = pred.eq(t_target.view_as(pred))
            acc = (correct.sum() / t_target.numel()).cpu().data.numpy()
        except AttributeError as e:
            acc = [get_acc(o, t) for o, t in zip(output, t_target)]

    return np.array(acc)


def train_community(
    model,
    train_loader,
    test_loader,
    optimizer,
    config,
    schedulers=None,
    n_epochs=None,
    use_tqdm=True,
    trials=(True, True),
    show_all_acc=False,
    stop_acc=None,
    device="cuda",
):
    n_epochs = config["training"]["n_epochs"] if n_epochs is None else n_epochs
    task = config["training"]["task"]
    decision = config["decision"]
    n_classes_per_digit = config["data"]["n_classes_per_digit"]

    descs = ["" for _ in range(2)]
    desc = lambda descs: descs[0] + descs[1]
    train_losses, train_accs = [], []
    test_accs, test_losses = [], []
    deciding_agents = []
    best_loss, best_acc = 1e10, 0.0
    training, testing = trials
    pbar = range(n_epochs + 1)

    tqdm_f = tqdm_n if is_notebook() else tqdm
    if use_tqdm:
        pbar = tqdm_f(pbar, position=0, leave=None, desc="Train Epoch:")

    torch.compile(model)

    for epoch in pbar:
        if training and epoch > 0:
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if type(data) is list:
                    data, target = [d.to(device) for d in data], target.to(device)
                else:
                    data, target = data.to(device), target.to(device)

                # Forward pass

                # Task Selection
                data, _ = process_data(data, config["data"])
                t_target = get_task_target(target, task, n_classes_per_digit)

                optimizer.zero_grad()

                output, _ = model(data)
                if decision is not None:
                    output, deciding_ags = get_decision(output, *decision)
                    both = decision[1] == "both"
                else:
                    deciding_ags = None
                    both = False

                try:
                    if (
                        deciding_ags is not None
                        and train_loader.batch_size in deciding_ags.shape
                    ):
                        deciding_agents.append(deciding_ags.cpu().data.numpy())
                except AttributeError:
                    deciding_ags = None

                complete_loss = get_loss(output, t_target, use_both=both)
                loss = nested_mean(complete_loss)

                acc = get_acc(output, t_target, use_both=both)

                train_accs.append(acc)

                loss.backward()
                train_losses.append(loss.cpu().data.item())

                # Apply gradients on agents weights
                optimizer.step()
                descs[0] = str(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.2f}, Accuracy: {}, Dec : {:.3f}%".format(
                        epoch,
                        batch_idx * train_loader.batch_size,
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        torch.round(complete_loss.mean(-1), decimals=3).data
                        if False
                        else torch.round(loss, decimals=1).item(),
                        np.round(100 * np.mean(acc))
                        if not show_all_acc
                        else nested_round(acc),
                        np.mean(deciding_agents),
                    )
                )

                if use_tqdm:
                    pbar.set_description(desc(descs))

        if testing:
            descs[1], loss, acc, _ = test_community(
                model,
                device,
                test_loader,
                config,
                show_all_acc,
            )
            if loss < best_loss:
                best_loss = loss
                best_state = copy.deepcopy(model.state_dict())
                best_acc = acc

            test_losses.append(loss)
            test_accs.append(acc)

            if use_tqdm:
                pbar.set_description(desc(descs))

        if schedulers is not None:
            for sch in schedulers:
                if sch:
                    sch.step()

        results = {
            "train_losses": np.array(train_losses),
            "train_accs": np.array(train_accs),
            "test_losses": np.array(test_losses),
            "test_accs": np.array(test_accs),
            "deciding_agents": np.array(deciding_agents),
            "best_state": best_state,
        }

        if stop_acc is not None and best_acc >= stop_acc:
            return results

    return results


def test_community(model, device, test_loader, config, show_all_acc=False):
    model.eval()
    test_loss = 0
    correct = 0
    acc = 0
    deciding_agents = []

    decision = config["decision"]
    task = config["training"]["task"]
    n_classes_per_digit = config["data"]["n_classes_per_digit"]

    with torch.no_grad():
        for data, target in test_loader:
            data, _ = process_data(data, config["data"])
            t_target = get_task_target(target, task, n_classes_per_digit)
            data, target, t_target = (
                data.to(device),
                target.to(device),
                t_target.to(device),
            )
            output, _ = model(data)

            if decision is not None:
                output, deciding_ags = get_decision(output, *decision)
                both = decision[1] == "both"
            else:
                deciding_ags = None
                both = False

            try:
                if (
                    deciding_ags is not None
                    and test_loader.batch_size in deciding_ags.shape
                ):
                    deciding_agents.append(deciding_ags.cpu().data.numpy())
            except AttributeError:
                deciding_ags = None

            complete_loss = get_loss(output, t_target, use_both=both)
            loss = nested_mean(complete_loss)

            test_loss += loss
            test_acc = get_acc(output, t_target, use_both=both)
            acc += test_acc

    test_loss /= len(test_loader)
    acc /= len(test_loader)

    deciding_agents = np.array(deciding_agents)

    desc = str(
        " | Test set: Loss: {:.3f}, Accuracy: {}%".format(
            test_loss,
            np.round(100 * np.mean(acc)) if not show_all_acc else nested_round(acc),
        )
    )

    return desc, test_loss.cpu().data.item(), acc, deciding_agents
