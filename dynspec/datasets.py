import torch
from torchvision import transforms
import numpy as np
from typing import Any, Callable, Tuple, Optional
from torchvision.datasets import EMNIST, MNIST
from PIL import Image
from torch.utils.data import Dataset
import os

"""
Custom Datasets
"""


def data_transform(img, transform):
    """
    Applies the given transform to the input image.

    Args:
        img (numpy.ndarray): The input image.
        transform (callable or None): The transform to apply to the image.

    Returns:
        The transformed image.
    """
    if transform is not None:
        img = Image.fromarray(img, mode="L")
        img = transform(img)
    return img


def target_transform(target, transform):
    """
    Applies a transformation function to the target tensor, if provided.

    Args:
        target (torch.Tensor): The target tensor to be transformed.
        transform (callable): The transformation function to be applied to the target tensor.

    Returns:
        torch.Tensor: The transformed target tensor.
    """
    if transform is not None:
        target = torch.tensor(transform(target))
    return target


class Custom_MNIST(MNIST):
    """
    Initializes a new custom MNIST dataset.

    Args:
        root (str): The root directory of the dataset.
        train (bool, optional): Whether to load the training or test split. Defaults to True.
        transform (Callable, optional): A function/transform that takes in a sample and returns a transformed version. Defaults to None.
        target_transform (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
        download (bool, optional): Whether to download the dataset if it is not already present. Defaults to False.
        truncate (list or int, optional): If not None, truncates the dataset to the specified number of classes. Defaults to None.
        pre_transform (bool, optional): Whether to apply the dataset's pre-processing transform. Defaults to True.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        truncate: list or int = None,
        pre_transform: bool = True,
    ) -> None:
        self.truncate = np.array(truncate) if truncate is not None else truncate
        self.pre_transform = pre_transform
        super().__init__(root, train, transform, target_transform, download)

    def _load_data(self):
        data, targets = super()._load_data()
        if self.truncate is not None:
            try:
                truncate_mask = list(
                    map(lambda t: t in self.truncate, np.array(targets))
                )
                self.truncate_values = np.sort(self.truncate)

            except ValueError:
                truncate_mask = np.array(targets) < self.truncate
                self.truncate_values = np.arange(self.truncate)

            data, targets = data[truncate_mask], targets[truncate_mask]

        else:
            self.truncate_values = np.arange(10)

        self.n_classes = len(self.truncate_values)

        if self.pre_transform:
            # We pre-process data for speed at training time. No random transforms are used here

            if self.transform is not None:
                data = data.numpy()
                v_transform = np.vectorize(
                    lambda d: data_transform(d, self.transform),
                    signature="(n, n) -> (n, n)",
                )
                data = v_transform(data)
                data = torch.from_numpy(data)

            if self.target_transform is not None:
                targets = torch.stack(
                    [target_transform(t, self.target_transform) for t in targets]
                )

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        if not self.pre_transform:
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)


class Custom_EMNIST(EMNIST):
    """
    Initializes a new custom EMNIST dataset.

    Args:
        root (str): The root directory of the dataset.
        train (bool, optional): Whether to load the training or test split. Defaults to True.
        transform (Callable, optional): A function/transform that takes in a sample and returns a transformed version. Defaults to None.
        target_transform (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
        download (bool, optional): Whether to download the dataset if it is not already present. Defaults to False.
        truncate (list or int, optional): If not None, truncates the dataset to the specified number of classes. Defaults to None.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        data_type: str = "digits",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        truncate: list = None,
    ) -> None:
        self.truncate = truncate
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            split=data_type,
        )

    def _load_data(self):
        data, targets = super()._load_data()
        if self.split == "letters":
            targets -= 1
        self.n_classes = targets.unique().shape[0]
        if self.truncate is not None:
            try:
                truncate_mask = np.array(targets) < self.truncate
                truncate_values = np.arange(self.truncate)
            except ValueError:
                truncate_mask = list(
                    map(lambda t: t in self.truncate, np.array(targets))
                )
                truncate_values = np.sort(self.truncate)
                truncate_values.sort()

            data, targets = data[truncate_mask], targets[truncate_mask]

            for i, t in enumerate(truncate_values):
                targets[targets == t] = i

            self.truncate_values = truncate_values

        self.n_classes = targets.unique().shape[0]

        if self.transform is not None:
            data = data.numpy()
            v_transform = np.vectorize(
                lambda d: data_transform(d, self.transform),
                signature="(n, n) -> (n, n)",
            )
            data = v_transform(data)
            data = torch.from_numpy(data)
        if self.target_transform is not None:
            targets = torch.stack(
                [target_transform(t, self.target_transform) for t in targets]
            )

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        return img, target


class DoubleDataset(Dataset):
    """
    Double Digits dataset
    Args :
        fix_asym : solve parity task asymetry by removing digit_1==digit_2 cases
        permute : permute targets classes
        seed : seed for permutation
        cov_ratio : change the covariance of the two targets, basically probability of t1 == t2
        transform : transform to apply to the images
    """

    def __init__(
        self,
        datasets,
        fix_asym=True,
        permute=False,
        seed=None,
        cov_ratio=1,
    ):
        super().__init__()

        self.datasets = datasets
        assert len(np.unique([d.n_classes for d in datasets])) == 1
        assert len(np.unique([len(d) for d in datasets])) == 1

        self.n_classes = datasets[0].n_classes

        self.fix_asym = fix_asym
        self.permute = permute
        if self.permute:
            if seed is None:
                seed = 42

            torch.manual_seed(seed)
            self.permutation1 = torch.randperm(self.n_classes)
            torch.manual_seed(seed + 1)
            self.permutation2 = torch.randperm(self.n_classes)

            self.permutations = [self.permutation1, self.permutation2]
            # print(self.permutations)
        else:
            self.permutations = [
                torch.arange(self.n_classes),
                torch.arange(self.n_classes),
            ]

        self.cov_ratio = cov_ratio
        self.create_all_idxs()

        self.data = self.create_data()

    def create_all_idxs(self):
        targets = [p[d.targets] for d, p in zip(self.datasets, self.permutations)]

        sorted_idxs = np.argsort(targets[0])

        t_idxs = [torch.where(targets[1] == t)[0] for t in range(self.n_classes)]
        c_idxs = [torch.where(targets[1] != t)[0] for t in range(self.n_classes)]

        idxs = [np.concatenate((t_idx, c_idx)) for t_idx, c_idx in zip(t_idxs, c_idxs)]

        ps = np.stack(
            [
                np.concatenate(
                    (np.ones_like(t_idx), self.cov_ratio * np.ones_like(c_idx))
                )
                for t_idx, c_idx in zip(t_idxs, c_idxs)
            ],
            -1,
        ).astype(float)
        ps /= ps.sum(0)

        self.cov_idxs = np.concatenate(
            [
                np.random.choice(idxs[t], size=len(t_idx), p=ps[:, t])
                for t, t_idx in enumerate(t_idxs)
            ]
        )[np.argsort(sorted_idxs)]

        # print((targets[0] == targets[1][self.cov_idxs]).float().mean())

        if self.fix_asym:
            self.asym_idxs = self.get_asym_indexs()
        else:
            self.asym_idxs = np.arange(len(self.datasets[0]))

    def valid_idx(self, idx):
        idx1, idx2 = idx, self.cov_idxs[idx]
        _, target_1 = self.datasets[0][idx1]
        _, target_2 = self.datasets[1][idx2]

        return not (
            (target_1 == target_2) or (target_1 == (target_2 - 1) % self.n_classes)
        )

    def get_asym_indexs(self):
        asym_idxs = []
        for idx in range(len(self.datasets[0])):
            if self.valid_idx(idx):
                asym_idxs.append(idx)
        return asym_idxs

    def __getitem__(self, index):
        digits, targets = [d[index] for d in self.data]

        return digits, targets

    def __len__(self):
        return len(self.asym_idxs)

    def create_data(self):
        # self.img_data = [
        #     torch.stack([d for d in dataset.data]) for dataset in self.datasets
        # ]
        # self.img_data = torch.stack([d.data for d in self.datasets], 1)

        # Filter data with pre-computed idxs
        data = [
            [d[torch.tensor(idx)] for i, d in enumerate([img_data, targets])]
            for idx, img_data, targets in zip(
                [self.asym_idxs, self.cov_idxs[self.asym_idxs]],
                [d.data for d in self.datasets],
                [d.targets for d in self.datasets],
            )
        ]

        data = [torch.stack(d, 1).squeeze() for i, d in enumerate(zip(*data))]

        if self.permute:
            data[1] = torch.stack(
                [self.permutations[i][t] for i, t in enumerate(data[1].T)], -1
            )
        data[0] = data[0].float()
        return data


def get_datasets(root, data_config):
    batch_size = data_config.get("batch_size", 128)
    data_sizes = data_config.get("data_size", None)
    use_cuda = data_config.get("use_cuda", torch.cuda.is_available())
    n_classes = data_config.get("n_classes_per_digit", 10)

    split_classes = data_config.get("split_classes", False)
    fix_asym = data_config.get("fix_asym", False)
    permute = data_config.get("permute_dataset", False)
    seed = data_config.get("seed", 42)
    cov_ratio = data_config.get("cov_ratio", 1.0)

    train_kwargs = {"batch_size": batch_size, "shuffle": True, "drop_last": True}
    test_kwargs = {"batch_size": batch_size, "shuffle": False, "drop_last": True}
    if use_cuda:
        cuda_kwargs = {"num_workers": 4, "pin_memory": True}

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform_digits = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    transform_letters = transforms.Compose(
        [
            lambda img: transforms.functional.rotate(img, -90),
            lambda img: transforms.functional.hflip(img),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    truncate_digits = np.arange(n_classes)

    kwargs = train_kwargs, test_kwargs

    if "digits" in data_config["data_type"]:
        single_digits = [
            Custom_MNIST(
                root,
                train=t,
                download=True,
                transform=transform_digits,
                truncate=truncate_digits,
            )
            for t in [True, False]
        ]

        double_digits = [
            DoubleDataset(
                [d, d],
                fix_asym=fix_asym,
                permute=permute,
                seed=seed,
                cov_ratio=cov_ratio,
                # transform=transform_digits,
            )
            for d in single_digits
        ]
    else:
        single_digits = [None, None]
        double_digits = [None, None]

    if "letters" in data_config["data_type"]:
        truncates = np.arange(10, 47)
        excludes = [18, 19, 21]  # exclude I, J, L
        for e in excludes:
            truncates = truncates[truncates != e]
        # np.random.shuffle(truncates)
        if split_classes:
            assert n_classes <= 17, "Max 17 classes for a separate set for each modules"
            truncates = truncates[: n_classes * 2]
            truncates = np.split(truncates, 2)
        else:
            assert n_classes <= 34, "Max 34 classes for the same set for each modules"
            truncates = truncates[:n_classes]
            truncates = truncates, truncates

        single_letters = [
            [
                Custom_EMNIST(
                    root,
                    train=t,
                    data_type="balanced",
                    truncate=trunc,
                    transform=transform_letters,
                )
                for t in [True, False]
            ]
            for trunc in truncates
        ]

        double_letters = [
            DoubleDataset(
                [s1, s2],
                fix_asym=fix_asym,
                permute=permute,
                seed=seed,
                cov_ratio=cov_ratio,
                # transform=transform_letters,
            )
            for (s1, s2) in zip(single_letters[0], single_letters[1])
        ]
    else:
        single_letters = [None, None]
        double_letters = [None, None]

    datasets = [
        single_digits,
        double_digits,
        single_letters,
        double_letters,
    ]
    if data_sizes is not None:
        datasets = [
            [
                (
                    torch.utils.data.Subset(d, torch.arange(d_size))
                    if d is not None
                    else None
                )
                for d, d_size in zip(dsets, data_sizes)
            ]
            for dsets in datasets
        ]

    loaders = [
        [
            torch.utils.data.DataLoader(d, **k) if d is not None else None
            for d, k in zip(dsets, kwargs)
        ]
        for dsets in datasets
    ]

    return {
        k: [dataset, loader]
        for k, dataset, loader in zip(
            [
                "single_digits",
                "double_digits",
                "single_letters",
                "double_letters",
            ],
            datasets,
            loaders,
        )
    }
