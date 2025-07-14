"""
Bits of code copied from equiadapt (https://github.com/arnab39/equiadapt).
All credits to the original authors.
This method simply avoids the need to install equiadapt as a dependency.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
import torchvision
from torchvision import transforms
import copy
import os


class SAMModel(nn.Module):
    def __init__(
        self,
        architecture_type: str,
        pretrained_ckpt_path: str,  # Segment-Anything Model requires a pretrained checkpoint path
    ):
        super().__init__()
        if not os.path.exists(pretrained_ckpt_path):
            raise FileNotFoundError(
                f"Pretrained checkpoint not found at {pretrained_ckpt_path}. Please download the SAM model checkpoint using the command: `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth`"
            )

        self.model = sam_model_registry[architecture_type](
            checkpoint=pretrained_ckpt_path
        )
        self.predictor = SamPredictor(self.model)

    def forward(self, images, targets):
        if isinstance(images, list):
            images = torch.stack(images)
        _, _, H, W = images.shape

        image_embeddings = self.model.image_encoder(images.cuda())
        pred_masks: List[torch.Tensor] = []
        ious: List[torch.Tensor] = []
        outputs: List[Dict[str, torch.Tensor]] = []
        for _, embedding, target in zip(images, image_embeddings, targets):
            bbox = target["boxes"]

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox.cuda(),
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))  # bbox_length x H x W
            ious.append(iou_predictions)  # bbox_length x 1

            output = dict(
                masks=torch.as_tensor(masks.squeeze(1) > 0.5, dtype=torch.uint8),
                scores=torch.as_tensor(iou_predictions.squeeze(1), dtype=torch.float32),
                labels=torch.as_tensor(target["labels"], dtype=torch.int64),
                boxes=torch.as_tensor(target["boxes"], dtype=torch.float32),
            )
            outputs.append(output)

        return None, pred_masks, ious, outputs


def rotate_image_and_target(tensor_image, target, angle):
    with torch.no_grad():
        # rotate the image and targets by angle degrees
        rotated_tensor_image = torchvision.transforms.functional.rotate(
            tensor_image, angle.item()
        )
        rotated_target = copy.deepcopy(target)

        rotated_target[0]["boxes"] = rotate_boxes(
            rotated_target[0]["boxes"], -angle, width=rotated_tensor_image.shape[-1]
        )
        rotated_target[0]["masks"] = rotate_masks(
            rotated_target[0]["masks"], angle.item()
        )

    return rotated_tensor_image, rotated_target


def rotate_masks(masks: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rotates masks by a specified angle.

    Args:
        masks (torch.Tensor): The masks to rotate.
        angle (torch.Tensor): The angle to rotate the masks by.

    Returns:
        torch.Tensor: The rotated masks.
    """
    return transforms.functional.rotate(masks, angle)


def rotate_points(
    origin: List[float], point: torch.Tensor, angle: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotates a point around an origin by a specified angle.

    Args:
        origin (List[float]): The origin to rotate the point around.
        point (torch.Tensor): The point to rotate.
        angle (torch.Tensor): The angle to rotate the point by.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The rotated point.
    """
    ox, oy = origin
    px, py = point

    qx = ox + torch.cos(angle) * (px - ox) - torch.sin(angle) * (py - oy)
    qy = oy + torch.sin(angle) * (px - ox) + torch.cos(angle) * (py - oy)
    return qx, qy


def rotate_boxes(boxes: torch.Tensor, angle: torch.Tensor, width: int) -> torch.Tensor:
    """
    Rotates bounding boxes by a specified angle.

    Args:
        boxes (torch.Tensor): The bounding boxes to rotate.
        angle (torch.Tensor): The angle to rotate the bounding boxes by.
        width (int): The width of the image.

    Returns:
        torch.Tensor: The rotated bounding boxes.
    """
    # rotate points
    origin: List[float] = [width / 2, width / 2]
    x_min_rot, y_min_rot = rotate_points(origin, boxes[:, :2].T, torch.deg2rad(angle))
    x_max_rot, y_max_rot = rotate_points(origin, boxes[:, 2:].T, torch.deg2rad(angle))

    # rearrange the max and mins to get rotated boxes
    x_min_rot, x_max_rot = (
        torch.min(x_min_rot, x_max_rot),
        torch.max(x_min_rot, x_max_rot),
    )
    y_min_rot, y_max_rot = (
        torch.min(y_min_rot, y_max_rot),
        torch.max(y_min_rot, y_max_rot),
    )
    rotated_boxes = torch.stack([x_min_rot, y_min_rot, x_max_rot, y_max_rot], dim=-1)

    return rotated_boxes
