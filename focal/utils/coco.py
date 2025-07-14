"""
COCO Dataset utilities for object detection and instance segmentation tasks.

This module provides classes for loading and processing COCO format datasets,
including functionality for resizing and padding images, handling annotations,
and creating PyTorch data loaders.
"""

# Standard library imports
import os
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms: List[nn.Module]) -> None:
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PILToTensor(nn.Module):
    def forward(self, image, target=None):
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, image, target=None):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class ResizeAndPad:
    """Resize and pad images and targets to a square size.

    This transformer handles both the image and its corresponding annotations (masks and boxes).
    It first resizes the image maintaining aspect ratio, then pads to make it square.
    """

    def __init__(self, target_size: int = 512) -> None:
        """Initialize ResizeAndPad.

        Args:
            target_size: Target size for the longest side after resizing
        """
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)

    def __call__(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply resize and pad transformations.

        Args:
            image: Input image tensor (C, H, W)
            target: Dictionary containing masks and boxes annotations

        Returns:
            Tuple of (transformed image, transformed target dict)
        """
        # Convert masks to numpy for transformation
        masks = [mask.numpy() for mask in target["masks"]]
        bboxes = target["boxes"].numpy()

        # Resize image and masks
        _, og_h, og_w = image.shape
        image = self.transform.apply_image_torch(image.unsqueeze(0)).squeeze(0)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]

        # Pad to square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w, pad_h = (max_dim - w) // 2, (max_dim - h) // 2
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)

        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding box coordinates for the resized and padded image
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [
            [bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h]
            for bbox in bboxes
        ]

        target["masks"] = torch.stack(masks)
        target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
        return image, target


class COCODataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for COCO dataset.

    Handles data loading and preprocessing for COCO dataset with support for
    testing. Training is intentionally not implemented.
    """

    def __init__(self, hyperparams: Optional[Dict] = None):
        """Initialize COCODataModule.

        Args:
            hyperparams: Dictionary containing dataset configuration parameters
        """
        super().__init__()
        self.hyperparams = hyperparams or {
            "root_dir": "data/coco",
            "ann_dir": "data/coco/annotations",
            "img_size": 512,
            "val_batch_size": 4,
            "num_workers": 0,
        }

    def get_transform(self, train: bool = True):
        """Get image transformation pipeline.

        Args:
            train: Whether to return training or validation transforms

        Returns:
            Composed transformation pipeline
        """
        tr = [PILToTensor(), ConvertImageDtype(torch.float)]
        return Compose(tr)

    def collate_fn(
        self, batch: List[Tuple[torch.Tensor, Dict]]
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
        """Custom collate function for batching data.

        Args:
            batch: List of (image, target) tuples

        Returns:
            Tuple of (images list, targets list)
        """
        images = [x[0] for x in batch]
        targets = [x[1] for x in batch]
        return images, targets

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for different stages (fit/test).

        Args:
            stage: Current stage ('fit' or 'test')
        """
        if stage == "fit" or stage is None:
            raise NotImplementedError("Training is not implemented for this dataset")
        if stage == "test":
            self.test_dataset = COCODataset(
                root_dir=os.path.join(self.hyperparams["root_dir"], "val2017"),
                annotation_file=os.path.join(
                    self.hyperparams["ann_dir"], "instances_val2017.json"
                ),
                transform=self.get_transform(train=False),
            )
            print("Test dataset size:", len(self.test_dataset))

    def train_dataloader(self) -> None:
        """Training is not implemented."""
        raise NotImplementedError("Training is not implemented for this dataset")

    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        valid_loader = DataLoader(
            self.valid_dataset,
            self.hyperparams["val_batch_size"],
            shuffle=False,
            num_workers=self.hyperparams["num_workers"],
            collate_fn=self.collate_fn,
        )
        return valid_loader

    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        test_loader = DataLoader(
            self.test_dataset,
            self.hyperparams["val_batch_size"],
            shuffle=False,
            num_workers=self.hyperparams["num_workers"],
            collate_fn=self.collate_fn,
        )
        return test_loader


class COCODataset(Dataset):
    """PyTorch Dataset for COCO instance segmentation.

    Loads images and annotations from COCO format dataset, supporting
    instance segmentation tasks with masks and bounding boxes.
    """

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform: Optional[callable] = None,
    ):
        """Initialize COCODataset.

        Args:
            root_dir: Directory containing images
            annotation_file: Path to COCO format annotation file
            transform: Optional transform to be applied to images and targets
            sam_transform: Optional SAM-specific transform (currently unused)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out images without annotations
        self.image_ids = [
            image_id
            for image_id in self.image_ids
            if len(self.coco.getAnnIds(imgIds=image_id)) > 0
        ]

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[Image.Image, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get an image and its annotations.

        Args:
            idx: Index of the image to fetch

        Returns:
            Tuple of (image, target dict) where target contains boxes, masks, and metadata
        """
        # Load image
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Load and process annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []
        labels = []
        iscrowds = []
        image_ids = []
        areas = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Skip degenerate boxes (area <= 0 or width/height < 1)
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue

            # Convert XYWH format to XYXY format for bounding boxes
            bboxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            masks.append(self.coco.annToMask(ann))
            image_ids.append(ann["image_id"])
            areas.append(ann["area"])
            iscrowds.append(ann["iscrowd"])

        # Create target dictionary with all annotations
        target = {
            "boxes": torch.as_tensor(bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
            "image_id": torch.as_tensor(image_ids[0], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowds, dtype=torch.int64),
        }

        # Apply transforms if specified
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target
