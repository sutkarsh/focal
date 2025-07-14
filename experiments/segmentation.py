"""
Runs segmentation experiments with N angles.

This module performs cyclic alignment experiments on segmentation tasks,
evaluating model performance under various rotation conditions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataclasses import asdict, dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import tyro

from focal.utils.energy import (
    CLIPEnergyArgs,
    DiffusionEnergyArgs,
    diff_energy,
    uncond_clip_energy,
)
from focal.utils.classifiers import CLIPClassifier
from focal.utils.coco import COCODataModule, ResizeAndPad
from focal.utils.datasets import load_prompts
from focal.utils.diffusion import StableDiffusion
from focal.utils.equiadapt_seg_utils import SAMModel, rotate_image_and_target


@dataclass(frozen=True)
class Args:
    """Configuration for segmentation alignment experiments.

    This class defines parameters for running rotation experiments where images
    are randomly rotated and then aligned back using various methods.
    """

    num_angles: int = 4
    """Number of angles for rotation"""

    dataset: Literal["coco"] = "coco"
    """Dataset to use for experiments"""

    img_size: Literal[512, 1024] = 1024
    """Input image size"""

    N: int = -1
    """Number of samples to process (-1 for all samples)"""

    diffusion: DiffusionEnergyArgs = DiffusionEnergyArgs(
        step_min=50, step_max=150, step_stride=10, factor=0.67
    )
    """Diffusion energy arguments"""

    clip_energy: CLIPEnergyArgs = CLIPEnergyArgs(factor=0.54, logit_top_factor=-0.2)
    """Classifier energy arguments"""

    seed: int = 0
    """Random seed for reproducibility"""

    device: str = "cuda:0"
    """Device to run the experiment on"""

    diffusion_device: str = "cuda:1"
    """Device for diffusion model; some low-end GPUs may not support all models on the same device"""


def move_to_device(tensor_dict: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Move tensors in dictionary to specified device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in tensor_dict.items()
    }


def evaluate_img(
    img_: torch.Tensor, targets: List[Dict[str, torch.Tensor]], segmenter: nn.Module
) -> float:
    """Evaluate segmentation performance.

    Args:
        img_: Input image tensor
        targets: List of target dictionaries
        segmenter: Segmentation model

    Returns:
        mAP score
    """
    with torch.inference_mode():
        _, _, _, outputs = segmenter(img_, targets)
        # Move tensors to CPU
        targets = [move_to_device(target, "cpu") for target in targets]
        outputs = [move_to_device(output, "cpu") for output in outputs]

        # Calculate mAP
        metric = MeanAveragePrecision(iou_type="segm")
        metric.update(outputs, targets)

    # Note: This is technically incorrect as it computes mAP one image at a time,
    # but to the best of my knowledge, this is how PRLC evaluates mAP.
    # See: https://github.com/arnab39/equiadapt/blob/main/examples/images/segmentation/model.py
    # So we do the same to maintain consistency.
    return metric.compute()["map"].item()


def generate_stats(results: Dict[str, List[Any]]) -> str:
    """Generate statistics string from results.

    Args:
        results: Dictionary containing experiment results

    Returns:
        Formatted string of statistics
    """
    return (
        f"mAP: {np.mean(results['correct']) * 100:.1f}→"
        f"{np.mean(results['correct_after_rot']) * 100:.1f}→"
        f"{np.mean(results['correct_after_rot_and_realign']) * 100:.1f} | "
        f"Pose: {np.mean(results['pose_accuracy']) * 100:.1f}% "
        f"({np.mean(results['pose_dist']):.1f}°)"
    )


def save_results(results: Dict[str, Any], args: Args) -> None:
    """Save experiment results to JSON file."""
    print("Saving results...")
    results["args"] = asdict(args)

    save_dir = Path(f"results/{args.dataset}/seg")
    save_dir.mkdir(parents=True, exist_ok=True)

    json_filename = f"seg_seed{args.seed}.json"
    save_path = save_dir / json_filename

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")


def main() -> None:
    """Main function to run the segmentation experiments."""
    args = tyro.cli(Args)
    print(f"Arguments: {args}")

    # Setup environment
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize transforms and data
    resize_pad = ResizeAndPad(target_size=args.img_size)
    resize224_crop = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224)]
    )

    # Setup COCO dataset
    coco = COCODataModule(
        {
            "root_dir": "datasets/coco/images",
            "ann_dir": "datasets/coco/annotations",
            "img_size": args.img_size,
            "val_batch_size": 1,
            "num_workers": 0,
        }
    )

    coco.setup(stage="test")
    dataloader = coco.test_dataloader()

    # Load SAM
    segmenter = (
        SAMModel("vit_h", "./checkpoints/sam_vit_h_4b8939.pth").to(device).eval()
    )

    # Initialize alignment models
    use_diffusion = args.diffusion.factor > 0
    use_clip_energy = args.clip_energy.factor > 0

    if not (use_diffusion or use_clip_energy):
        raise ValueError(
            "Need to use at least one alignment method. "
            "Did you mean to set clip_energy.factor or diffusion.factor to a positive value?"
        )

    clip_model = None
    if use_clip_energy:
        with torch.inference_mode():
            prompts = load_prompts("coco")
            clip_model = CLIPClassifier(prompts, device=args.diffusion_device)

    diffusion_model = None
    if use_diffusion:
        diffusion_model = StableDiffusion(
            model_id=args.diffusion.model,
            size=args.diffusion.size,
            device=args.diffusion_device,
        )

    # Setup experiment parameters
    num_samples = len(dataloader) if args.N == -1 else args.N
    angles = np.linspace(-180, 180, args.num_angles, endpoint=False)

    # Initialize results
    results = {
        "correct": [],
        "correct_after_rot": [],
        "correct_after_rot_and_realign": [],
        "thetas": [],
        "pose_accuracy": [],
        "pose_dist": [],
        "inferred_angles": [],
    }

    # Run experiments
    pbar = tqdm(iter(dataloader), total=num_samples, dynamic_ncols=True)
    with torch.inference_mode():
        for idx, (im, targets) in enumerate(pbar):
            if idx >= num_samples:
                break

            # Select random angle
            theta = np.random.choice(angles)
            results["thetas"].append(theta)

            """Process images and targets for evaluation."""
            im_unpadded = resize224_crop(im[0].clone())

            assert len(targets) == 1, (
                "Only one target expected per image in COCO dataset"
            )

            # Apply resize and padding
            im, targets[0] = resize_pad(im[0].cpu(), move_to_device(targets[0], "cpu"))
            im = im.to(device).unsqueeze(0)
            targets = [move_to_device(targets[0], device)]

            # Evaluate default accuracy
            results["correct"].append(evaluate_img(im, targets, segmenter))

            # Rotate and evaluate
            im, targets = rotate_image_and_target(
                im, targets, torch.tensor(theta, device=device)
            )
            im_unpadded = transforms.functional.rotate(im_unpadded, theta)

            results["correct_after_rot"].append(evaluate_img(im, targets, segmenter))
            # Generate rotated versions
            rot_im_targ = [
                rotate_image_and_target(im, targets, torch.tensor(a, device=device))
                for a in angles
            ]

            rot_ims = torch.cat([im_ for im_, _ in rot_im_targ], dim=0)

            # Calculate alignment scores
            final_score = 0
            if use_clip_energy:
                final_score += args.clip_energy.factor * uncond_clip_energy(
                    rot_ims, clip_model, args.clip_energy
                )
            if use_diffusion:
                final_score += args.diffusion.factor * diff_energy(
                    rot_ims, diffusion_model, args.diffusion
                )

            # Find and apply best angle
            best_angle = angles[np.argmin(final_score)]
            im_realign, targets_realign = rotate_image_and_target(
                im, targets, torch.tensor(best_angle, device=device)
            )

            results["correct_after_rot_and_realign"].append(
                evaluate_img(im_realign, targets_realign, segmenter)
            )

            results["inferred_angles"].append(best_angle)

            # Calculate pose metrics
            best_angle_remapped = ((180 - best_angle) % 360) - 180
            results["pose_accuracy"].append(
                int(abs(theta - best_angle_remapped) < 1e-6)
            )

            theta_cossin = np.array(
                [np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))]
            )
            pred_cossin = np.array(
                [
                    np.cos(np.deg2rad(best_angle_remapped)),
                    np.sin(np.deg2rad(best_angle_remapped)),
                ]
            )
            cossim = np.clip(np.dot(theta_cossin, pred_cossin), -1, 1)
            results["pose_dist"].append(np.rad2deg(np.arccos(cossim)))

            # Update progress bar
            pbar.set_postfix_str(generate_stats(results))

    print(generate_stats(results))
    save_results(results, args)


if __name__ == "__main__":
    main()
