import torch
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm as tqdm_n
from tqdm.notebook import tqdm
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
        all = [get_acc(o, t_target) for o in output]
        acc, correct = [a[0] for a in all], [a[1] for a in all]
    else:
        try:
            pred = output.argmax(
                dim=-1, keepdim=True
            )  # get the index of the max log-probability

            correct = pred.eq(t_target.view_as(pred))
            acc = (correct.sum() / t_target.numel()).cpu().data.numpy()
            correct = correct.cpu().data.numpy().squeeze()
        except AttributeError as e:
            all = [get_acc(o, t) for o, t in zip(output, t_target)]
            acc, correct = [a[0] for a in all], [a[1] for a in all]

    return np.array(acc), np.array(correct)


def check_grad(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                print(f"No grad for {n}")
            elif not p.grad.any():
                print(f"Zero grad for {n}")
            else:
                pass


def train_community(
    model,
    optimizer,
    config,
    loaders,
    schedulers=None,
    n_epochs=None,
    use_tqdm=True,
    trials=(True, True),
    show_all_acc=False,
    stop_acc=None,
    device="cuda",
    pbar=None,
):
    n_epochs = config["training"]["n_epochs"] if n_epochs is None else n_epochs
    task = config["training"]["task"]
    if task == "parity-digits-both":
        task = config["training"]["task"] = ["parity-digits", "inv_parity-digits"]
        show_acc = False

    decision = config["decision"]
    n_classes_per_digit = config["data"]["n_classes_per_digit"]

    descs = ["" for _ in range(2 + (pbar is not None))]
    desc = lambda descs: np.array(descs, dtype=object).sum()
    train_losses, train_accs = [], []
    test_accs, test_losses, all_accs = [], [], []
    deciding_modules = []
    best_loss, best_acc = 1e10, 0.0
    training, testing = trials

    tqdm_f = tqdm_n if is_notebook() else tqdm
    if use_tqdm:
        pbar_e = range(n_epochs + 1)
        if pbar is None:
            pbar_e = tqdm_f(pbar_e, position=0, leave=None, desc="Train Epoch:")
            pbar = pbar_e
        else:
            descs[0] = pbar.desc

    # model = torch.compile(model)

    train_loader, test_loader = loaders
    for epoch in pbar_e:
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
                        deciding_modules.append(deciding_ags.cpu().data.numpy())
                except AttributeError:
                    deciding_ags = None

                complete_loss = get_loss(output, t_target, use_both=both)
                loss = nested_mean(complete_loss)

                acc, _ = get_acc(output, t_target, use_both=both)

                train_accs.append(acc)

                loss.backward()
                if config["training"].get("check_grad", False):
                    check_grad(model)
                train_losses.append(loss.cpu().data.item())

                if show_all_acc is True:
                    show_acc = nested_round(acc)
                elif type(show_all_acc) is int:
                    show_acc = nested_round(acc[show_all_acc])
                else:
                    show_acc = np.round(100 * np.mean(acc))

                # Apply gradients on modules weights
                optimizer.step()
                descs[-2] = str(
                    "Train Epoch: {}/{} [{}/{} ({:.0f}%)] Loss: {:.2f}, Accuracy: {} %".format(
                        epoch,
                        n_epochs,
                        batch_idx * train_loader.batch_size,
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        torch.round(complete_loss.mean(-1), decimals=3).data
                        if False
                        else torch.round(loss, decimals=1).item(),
                        show_acc,
                        # np.mean(deciding_modules) if len(deciding_modules) else None,
                    )
                )

                if use_tqdm:
                    pbar.set_description(desc(descs))

        if testing:
            test_results = test_community(
                model,
                device,
                test_loader,
                config,
                show_all_acc,
            )
            descs[-1] = test_results["desc"]

            if test_results["test_loss"] < best_loss:
                best_loss = test_results["test_loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_acc = test_results["test_acc"]

            test_losses.append(test_results["test_loss"])
            test_accs.append(test_results["test_acc"])
            all_accs.append(test_results["all_accs"])

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
            "test_accs": np.array(test_accs),  # n_epochs x n_tasks
            "all_accs": np.array(all_accs),  # n_epochs x n_tasks x n_data
            "deciding_modules": np.array(deciding_modules),
            "best_state": best_state,
        }

        try:
            if stop_acc is not None and best_acc >= stop_acc:
                return results
        except ValueError:
            if stop_acc is not None and (best_acc >= stop_acc).all():
                return results

    return results


def test_community(model, device, test_loader, config, show_all_acc=False):
    model.eval()
    test_loss = 0
    test_accs = []
    all_accs = []
    deciding_modules = []

    decision = config["decision"]
    task = config["training"]["task"]

    n_classes_per_digit = config["data"]["n_classes_per_digit"]

    with torch.no_grad():
        for data, target in test_loader:
            data, _ = process_data(data, config["data"])
            data, target = (
                data.to(device),
                target.to(device),
            )
            t_target = get_task_target(target, task, n_classes_per_digit)
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
                    deciding_modules.append(deciding_ags.cpu().data.numpy())
            except AttributeError:
                deciding_ags = None

            complete_loss = get_loss(output, t_target, use_both=both)
            loss = nested_mean(complete_loss)

            test_loss += loss
            test_acc, all_acc = get_acc(output, t_target, use_both=both)
            test_accs.append(test_acc)
            all_accs.append(all_acc)

    test_loss /= len(test_loader)

    acc = np.array(test_accs).mean(0)

    deciding_modules = np.array(deciding_modules)

    if show_all_acc is True:
        show_acc = nested_round(acc)
    elif type(show_all_acc) is int:
        show_acc = nested_round(acc[show_all_acc])
    else:
        show_acc = np.round(100 * np.mean(acc))

    desc = str(
        " | Test set: Loss: {:.3f}, Accuracy: {}%".format(
            test_loss,
            show_acc,
        )
    )

    test_results = {
        "desc": desc,
        "test_loss": test_loss.cpu().data.item(),
        "test_acc": acc,
        "deciding_modules": deciding_modules,
        "all_accs": np.concatenate(all_accs, -1),
    }

    return test_results
