"""Dataset utilities and loaders for various image classification tasks."""

import glob
import json
import os
import os.path as osp
import csv
from typing import Any, Dict, List, Optional, Tuple
import random

import pandas as pd
import torch
from PIL import Image
import torchvision
from torchvision import datasets
from torchvision.datasets import VisionDataset

import numpy as np
from torchvision import transforms
from omegaconf import DictConfig
from einops import rearrange

from .imagenet_classnames import get_classnames

# Constants
DATASET_ROOT = "./datasets"
IMAGENET_ROOT = "/home/utkarsh/data/imagenet"
PROMPTS_ROOT = "./focal/prompts"


def load_classes(datasetname: str) -> List[str]:
    """Load classes from CSV file.

    Args:
        datasetname: Name of the dataset

    Returns:
        List of classes

    Raises:
        FileNotFoundError: If prompts file doesn't exist
    """
    prompt_path = osp.join(PROMPTS_ROOT, f"{datasetname}_prompts.csv")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompts file not found: {prompt_path}")
    return list(pd.read_csv(prompt_path)["classname"].values)


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


def rotate_image_objaverse(
    image: Image.Image, image_alpha: Image.Image, angle: float
) -> Image.Image:
    """Rotate an image by a given angle.

    Args:
        image: The input RGB image to rotate, with pixels filled in with white. This image is for OVSeg predictions.
        image_alpha: The input RGBA channel image to rotate, with pixels filled in with 0 alpha. This image is for passing to TRELLIS.
        angle: The angle in degrees to rotate the image.

    Returns:
        The rotated image and the rotated alpha image.
    """
    rotated_img_alpha = image_alpha.rotate(
        angle, fillcolor=(0, 0, 0, 0), resample=Image.BILINEAR
    )
    rotated_img = image.rotate(
        angle, fillcolor=(255, 255, 255), resample=Image.BILINEAR
    )
    return rotated_img, rotated_img_alpha


class ObjaverseDataset(torch.utils.data.Dataset):
    """Objaverse dataset class."""

    def __init__(
        self,
        cfg: DictConfig,
        transform: Any = torchvision.transforms.ToTensor(),
        **kwargs,
    ) -> None:
        """Initialize the Objaverse dataset.

        Args:
            cfg: Configuration object containing dataset parameters.
            transform: Transformations to apply to the images.
        """
        super().__init__()
        self.root_dir = cfg.objaverse.dataset_root
        self.cfg = cfg
        self.transform = transform

        self.list_file = f"{cfg.objaverse.dataset_root}/filtered_objaverse_list.txt"
        with open(self.list_file) as f:
            self.objects = [line.strip() for line in f.readlines()]

        self.set_obja_locations(
            el_min=cfg.objaverse.el_min,
            el_max=cfg.objaverse.el_max,
            el_step=cfg.objaverse.el_step,
            az_min=cfg.objaverse.az_min,
            az_max=cfg.objaverse.az_max,
            az_step=cfg.objaverse.az_step,
            radius=cfg.objaverse.radius,
        )

        self.num_renders = len(self.el_list)

        # load class info
        import objaverse

        lvis_annotations = objaverse.load_lvis_annotations()
        self.classes = []
        for k in lvis_annotations.keys():
            self.classes.append(k)

        self.corrected_classes = [
            cls_name.replace(" ", "_") for cls_name in self.classes
        ]

        self.num_classes = len(self.corrected_classes)

    def set_obja_locations(
        self,
        el_min: int,
        el_max: int,
        el_step: int,
        az_min: int,
        az_max: int,
        az_step: int,
        radius: float,
    ) -> Tuple[List[int], List[int], List[float], List[float], List[float]]:
        """Return a grid of locations to use for base Objaverse input views.

        Args:
            el_min: start elevation
            el_max: end elevation
            el_step: elevation interval
            az_min: start azimuth
            az_max: end azimuth
            az_step: azimuth interval

        Returns:
            A tuple of 5 lists containing the list of elevations, azimuths, and then
            x, y, and z camera coordinates for Objaverse's blender script
        """
        self.el_list = []
        self.az_list = []
        self.x_list = []
        self.y_list = []
        self.z_list = []
        for el in range(el_min, el_max, el_step):
            for az in range(az_min, az_max, az_step):
                self.el_list.append(el)
                self.az_list.append(az)
                el_rad = el * np.pi / 180
                az_rad = az * np.pi / 180
                self.x_list.append(radius * np.sin(az_rad) * np.cos(el_rad))
                self.y_list.append(-radius * np.cos(az_rad) * np.cos(el_rad))
                self.z_list.append(radius * np.sin(el_rad))

    def __len__(self) -> int:
        return len(self.objects)

    def load_obja_render(self, path: str, return_type: str = "rgb") -> Image.Image:
        """Load an Objaverse rendered image at the given path.

        Args:
            path: the location of the image
            return_type: 'alpha' to return the image with alpha channel,
                         'rgb' to return the image with alpha channel removed

        Returns:
            The image in PIL format
        """
        image = Image.open(path)
        assert return_type in ["alpha", "rgb"], "return_type must be 'alpha' or 'rgb'"
        if return_type == "alpha":
            return image
        # Otherwise, composite it into an RGB image and return that instead
        image_np = np.array(image)
        alpha = image_np[:, :, 3:4] / 255.0
        white_im = np.ones_like(image_np) * 255.0
        image_np = alpha * image_np + (1.0 - alpha) * white_im
        image = Image.fromarray(image_np.astype(np.uint8)[:, :, :3])
        return image

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """Get item by index.

        Args:
            index: Dataset index

        Returns:
            Tuple of (transformed image, class label)
        """
        this_object = self.objects[index]
        label = self.corrected_classes.index(this_object.split("/")[3])
        imgs = []
        imgs_alpha = []
        for el, az, x, y, z in zip(
            self.el_list, self.az_list, self.x_list, self.y_list, self.z_list
        ):
            path = os.path.join(
                this_object,
                f"{el}_{az}_{x:.2f}_{y:.2f}_{z:.2f}.png",
            )
            image = self.load_obja_render(path)
            image_torch = self.transform(image)
            imgs.append(image_torch)
            image_alpha = self.load_obja_render(path, return_type="alpha")
            image_alpha_torch = self.transform(image_alpha)
            imgs_alpha.append(image_alpha_torch)

        img_tensor = torch.stack(imgs, dim=0)  # shape: (num_renders, C, H, W)
        img_alpha_tensor = torch.stack(
            imgs_alpha, dim=0
        )  # shape: (num_renders, C, H, W)
        return img_tensor, img_alpha_tensor, label


def rotate_image_co3d(
    image: Image.Image, image_black: Image.Image, angle: float
) -> Image.Image:
    """Rotate an image by a given angle.

    Args:
        image: The input RGB image to rotate with pixels filled in with white. To be used for OVSeg predictions.
        image_black: The input RGB image with black background to rotate. Pixels filled in with black, to be used in
                     as input to TRELLIS. Black background is used because we manually preprocess the image per TRELLIS'
                     preprocessing function, rather than calling it, because we have ground truth masks available.
        angle: The angle in degrees to rotate the image.

    Returns:
        The rotated image and the rotated black background image.
    """
    rotated_img_black = image_black.rotate(
        angle, fillcolor=(0, 0, 0), resample=Image.BILINEAR
    )
    rotated_img = image.rotate(
        angle, fillcolor=(255, 255, 255), resample=Image.BILINEAR
    )
    return rotated_img, rotated_img_black


class CO3DDataset(torch.utils.data.Dataset):
    """CO3D dataset class."""

    def __init__(
        self,
        cfg: DictConfig,
        transform: Any = torchvision.transforms.ToTensor(),
        thresh: float = 0.3,
        **kwargs,
    ) -> None:
        """Initialize the CO3D dataset.

        Args:
            root: Root directory of the dataset.
            transform: Transformations to apply to the images.
            thresh: Threshold value for filtering. This is the maximum threshold of GT probability
                    in filtered frames. Files must be processed according to the CO3D processing script.
        """
        super().__init__()
        self.root_dir = cfg.co3d.dataset_root
        self.cfg = cfg
        self.transform = transform

        self.input_file = cfg.co3d.input_file.replace("[thresh]", str(thresh))
        self.dataset_root = cfg.co3d.dataset_root.replace("[thresh]", str(thresh))
        self.json_file = cfg.co3d.json_file.replace("[thresh]", str(thresh))
        self.class_file = cfg.co3d.class_file.replace("[thresh]", str(thresh))

        # load class names
        with open(self.class_file) as f:
            self.classes = [line.strip() for line in f.readlines()]

        # load json saying which images have passed the filter or not
        with open(self.json_file, "r") as file:
            data = json.load(file)

        # build map from images that passed the filter to the index in the original list with all the images
        self.img_file_map = {}
        with open(self.input_file, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            self.rows = []
            for i, row in enumerate(reader):
                row_status = data[os.path.join(f"{self.dataset_root}", f"{i}.png")][
                    "status"
                ]
                assert row_status in ["PASS", "FAIL"], (
                    f"Invalid row status at index {i}: {row_status}"
                )
                if row_status == "PASS":
                    self.img_file_map[len(self.rows)] = i
                    self.rows.append(row)

        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """Get item by index.

        Args:
            index: Dataset index

        Returns:
            Tuple of (transformed image tensor with white background, transformed image tensor with black background, class label)
            Returned images are of size (518x518)
        """
        row = self.rows[index]

        this_cls_label = row[0].split("/")[3]
        assert this_cls_label in self.classes, (
            f"Index {index}: Class label {this_cls_label} not in self.classes"
        )
        label = self.classes.index(this_cls_label)

        image_crop = Image.open(f"{self.dataset_root}/{self.img_file_map[index]}.png")
        mask_crop = Image.open(f"{self.dataset_root}/{self.img_file_map[index]}s.png")

        img_np = np.array(image_crop)
        mask_np = np.array(mask_crop)
        foreground_mask = mask_np[:, :, None] > 128
        # create a black background version for TRELLIS and a white background version for OVSeg
        img_black_bg = np.where(foreground_mask, img_np, 0)
        img_white_bg = np.where(foreground_mask, img_np, 255)

        # convert torch, add extra view dimension
        img_white_bg = rearrange(
            self.transform(Image.fromarray(img_white_bg)), "c h w -> 1 c h w"
        )
        img_black_bg = rearrange(
            self.transform(Image.fromarray(img_black_bg)), "c h w -> 1 c h w"
        )
        return img_white_bg, img_black_bg, label


class ObjaverseMetaDataset:
    """Objaverse helper to download data and store metadata about the dataset."""

    def __init__(self, obj_path: str, download: bool):
        """
        Initialize Objaverse dataset.

        Args:
            obj_path: Path to object annotations
            download: Whether to download the data or not
        """
        import objaverse
        import gzip

        # load lvis uids
        lvis_annotations = objaverse.load_lvis_annotations()
        self.lvis_uids = []
        count = 0
        for k in lvis_annotations.keys():
            for uid in lvis_annotations[k]:
                count += 1
                self.lvis_uids.append(uid)

        # download objaverse data if necessary
        self.download_data(download, self.lvis_uids)

        # load annotations
        with gzip.open(obj_path, "rb") as f:
            self.uid_to_object_paths = json.load(f)

        self.uid_to_label_map = {}
        for k in lvis_annotations.keys():
            for uid in lvis_annotations[k]:
                self.uid_to_label_map[uid] = k

        # convert uid_to_label_map to list of keys
        self.uids_list = list(self.uid_to_label_map.keys())
        random.shuffle(self.uids_list)

        # load class list. Correct for monitor class to not have spaces
        self.classes = []
        for k in lvis_annotations.keys():
            self.classes.append(k)

        self.corrected_classes = self.classes.copy()
        self.corrected_classes[
            self.corrected_classes.index(
                "monitor_(computer_equipment) computer_monitor"
            )
        ] = "monitor_(computer_equipment)_computer_monitor"
        cls_map = {
            cls: corr_cls for cls, corr_cls in zip(self.classes, self.corrected_classes)
        }
        self.corrected_uid_to_label_map = {
            uid: cls_map[label] for uid, label in self.uid_to_label_map.items()
        }

        self.text_annos = []
        for cls in self.classes:
            self.text_annos.append("a photo of a " + cls)

    def download_data(self, download: bool, lvis_uids: List[str]) -> None:
        """Function to download Objaverse data if necessary."""

        """
        Args:
            download: whether to download the data or not
        """
        if download:
            import objaverse
            import multiprocessing

            processes = multiprocessing.cpu_count()

            objaverse.load_objects(uids=lvis_uids, download_processes=processes)
