"""Dataset utilities and loaders for various image classification tasks."""

import glob
import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torchvision import datasets
from torchvision.datasets import VisionDataset

import numpy as np
from torchvision import transforms

from .imagenet_classnames import get_classnames

# Constants
DATASET_ROOT = "./datasets"
IMAGENET_ROOT = "/home/utkarsh/data/imagenet"
PROMPTS_ROOT = "./focal/prompts"


def load_prompts(datasetname: str) -> List[str]:
    """Load text prompts from CSV file.

    Args:
        datasetname: Name of the dataset

    Returns:
        List of text prompts

    Raises:
        FileNotFoundError: If prompts file doesn't exist
    """
    prompt_path = osp.join(PROMPTS_ROOT, f"{datasetname}_prompts.csv")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompts file not found: {prompt_path}")
    return list(pd.read_csv(prompt_path)["prompt"].values)


def get_classes_templates(dataset: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """Get class names and templates for text prompts.

    Args:
        dataset: Dataset name

    Returns:
        Tuple of (class dictionary, templates list)

    Raises:
        NotImplementedError: If dataset is not supported
        ValueError: If dataset lacks required entries
    """
    template_path = os.path.join(os.path.dirname(__file__), "templates.json")
    with open(template_path, "r") as f:
        all_templates = json.load(f)

    if dataset not in all_templates:
        raise NotImplementedError(
            f"Dataset {dataset} not implemented. Supported: {list(all_templates.keys())}"
        )

    entry = all_templates[dataset]
    if "classes" not in entry or "templates" not in entry:
        raise ValueError(f"Dataset {dataset} missing required entries")

    classes_dict, templates = entry["classes"], entry["templates"]
    if isinstance(classes_dict, list):
        classes_dict = {c: [c] for c in classes_dict}

    return classes_dict, templates


# Load ImageNet class indices
with open(
    os.path.join(os.path.dirname(__file__), "imagenet_class_index.json"), "r"
) as f:
    IMAGENET_CLASS_INDEX = json.load(f)
FOLDER_TO_CLASS = {folder: int(i) for i, (folder, _) in IMAGENET_CLASS_INDEX.items()}


class MNIST(datasets.MNIST):
    """MNIST dataset with string-based class indices."""

    class_to_idx = {str(i): i for i in range(10)}


def get_target_dataset(
    name: str,
    train: bool = False,
    transform: Optional[Any] = None,
    target_transform: Optional[Any] = None,
) -> VisionDataset:
    """Get dataset by name with consistent attributes.

    Args:
        name: Dataset name
        train: Whether to use training split
        transform: Transform to apply to images
        target_transform: Transform to apply to targets

    Returns:
        Dataset with class_to_idx and file_to_class attributes

    Raises:
        ValueError: If dataset is not supported or lacks train split
        AssertionError: If dataset lacks required attributes
    """
    dataset_map = {
        "cifar10": (datasets.CIFAR10, {"train": train}),
        "cifar100": (datasets.CIFAR100, {"train": train}),
        "stl10": (datasets.STL10, {"split": "train" if train else "test"}),
    }

    if name in dataset_map:
        dataset_cls, extra_args = dataset_map[name]
        dataset = dataset_cls(
            root=DATASET_ROOT,
            transform=transform,
            target_transform=target_transform,
            download=True,
            **extra_args,
        )
    elif name == "imagenet":
        if train:
            dataset = datasets.ImageNet(
                root=IMAGENET_ROOT,
                split="train",
                transform=transform,
                target_transform=target_transform,
            )
        else:
            dataset = datasets.ImageFolder(
                root=osp.join(DATASET_ROOT, "./imgnet_testset"),
                transform=transform,
                target_transform=target_transform,
            )
            dataset.class_to_idx = None
            dataset.classes = get_classnames("openai")
            dataset.file_to_class = None
    else:
        raise ValueError(f"Dataset {name} not supported")

    # Post-processing for specific datasets
    if name == "stl10":
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
    elif name in {"cifar10", "cifar100", "stl10"}:
        dataset.file_to_class = {
            str(idx): dataset[idx][1] for idx in range(len(dataset))
        }

    return dataset


class DayNightDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset_dir="/home/utkarsh/urbanir/results_valid", transform=None
    ):
        import glob

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.day_files = sorted(glob.glob(f"{dataset_dir}/day/*.png"))
        self.night_files = [f.replace("day", "night") for f in self.day_files]
        self.files = list(zip(self.day_files, self.night_files))
        np.random.shuffle(self.files)

        print(f"Found {len(self.files)} day-night pairs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        day_file, night_file = self.files[idx]
        day_img = transforms.ToTensor()(
            transforms.ToPILImage()(np.array(Image.open(day_file)))
        )
        night_img = transforms.ToTensor()(
            transforms.ToPILImage()(np.array(Image.open(night_file)))
        )

        day_img = transforms.Resize(512)(day_img)
        night_img = transforms.Resize(512)(night_img)

        day_img = transforms.CenterCrop(512)(day_img)
        night_img = transforms.CenterCrop(512)(night_img)

        assert day_img.shape == night_img.shape == (3, 512, 512), (
            f"Invalid shape: {day_img.shape}, {night_img.shape}"
        )

        if self.transform:
            day_img = self.transform(day_img)
            night_img = self.transform(night_img)

        return day_img, night_img
