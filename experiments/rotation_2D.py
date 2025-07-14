"""
rotation_2D.py
Runs 2D rotation experiments with N angles.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode, rotate as torch_rotate
from tqdm import tqdm
import tyro
from dataclasses import dataclass, asdict

from focal.utils.datasets import get_target_dataset, load_prompts
from focal.utils.classifiers import (
    CLIPClassifier,
    DINOv2Classifier,
    ResNet50,
    ViTB,
    PRLC_R50,
    PRLC_ViTB,
)
from focal.utils.diffusion import StableDiffusion
from focal.utils.energy import (
    DiffusionEnergyArgs,
    CLIPEnergyArgs,
    diff_energy,
    uncond_clip_energy,
)


@dataclass(frozen=True)
class Args:
    """Configuration for C-8 Alignment Classifier experiments.

    This class defines parameters for running rotation experiments where images
    are randomly rotated and then aligned back using various methods.
    """

    num_angles: int = 8
    """Number of angles for rotation"""

    dataset: Literal["cifar10", "cifar100", "stl10", "imagenet"] = "cifar10"
    """Dataset to use for experiments"""

    model: Literal["clip", "resnet", "vitb", "dino", "prlc_r50", "prlc_vit"] = "clip"
    """Model architecture for downstream classification"""

    ckpt: Union[str, None] = None
    """Path to the pretrained checkpoint for the model; only used for PRLC models"""

    N: int = -1
    """Number of samples to process (-1 for all samples)"""

    diffusion: DiffusionEnergyArgs = DiffusionEnergyArgs()
    """Diffusion energy arguments"""

    clip_energy: CLIPEnergyArgs = CLIPEnergyArgs(factor=1.0)
    """Classifier energy arguments"""

    seed: int = 0
    """Random seed for reproducibility"""

    device: str = "cuda:0"
    """Device to run the experiment on"""


def resize_crop(im: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Resize and center crop an image tensor to the specified size.

    Args:
        im: Input image tensor
        size: Desired output size (default is 224)
    Returns:
        Resized and cropped image tensor of size (size, size)
    """
    im = transforms.Resize(224)(im)
    im = transforms.CenterCrop(224)(im)
    return im


def rotate(im: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate an image by the given angle.

    Args:
        im: Input image tensor, shape (N, C, H, W)
        angle: Rotation angle in degrees

    Returns:
        Rotated image tensor of size 224x224
    """
    assert im.dim() == 4, "Input image must be a 4D tensor (N, C, H, W)"
    assert im.size(1) == 3, (
        f"Input image must have 3 channels (RGB), got shape {im.shape}"
    )

    padding = int(
        224 * 0.4
    )  # Excessively large padding to ensure no cropping after rotation
    impad = transforms.Pad(padding, padding_mode="edge")(im)
    impadrot = torch_rotate(impad, angle, interpolation=InterpolationMode.BILINEAR)
    return transforms.CenterCrop(224)(impadrot)


def evaluate_img(
    img: torch.Tensor,
    classifier: Union[CLIPClassifier, ResNet50, ViTB, DINOv2Classifier],
    label: int,
) -> int:
    """Evaluate if classifier prediction matches the label.

    Args:
        img: Input image tensor, shape (1, 3, H, W)
        classifier: Classifier model
        label: Ground truth label

    Returns:
        1 if prediction is correct, 0 otherwise
    """
    assert img.dim() == 4 and img.size(0) == 1, (
        "Input image must be a 4D tensor with batch size 1"
    )
    assert img.size(1) == 3, (
        f"Input image must have 3 channels (RGB), got shape {img.shape}"
    )
    pred = classifier(img)
    pred_label = pred.argmax().cpu().item()
    return int(label == pred_label)


def generate_stats(results: Dict[str, List[Any]]) -> str:
    """Generate statistics string from results.

    Args:
        results: Dictionary containing experiment results

    Returns:
        Formatted string of statistics
    """
    stats = [
        f"Default: {np.mean(results['correct']) * 100:.1f}%",
        f"Rot: {np.mean(results['correct_after_rot']) * 100:.1f}%",
        f"Rot+Unrot: {np.mean(results['correct_after_rot_and_unrot']) * 100:.1f}%",
        f"Rot+Realign: {np.mean(results['correct_after_rot_and_realign']) * 100:.1f}%",
        f"Upright+Realign: {np.mean(results['correct_upright_align']) * 100:.3f}%",
        f"Pose Acc: {np.mean(results['pose_accuracy']) * 100:.3f}%",
        f"Pose Dist: {np.mean(results['pose_dist']):.2f}",
    ]
    return " ".join(stats)


def save_results(results: Dict[str, Any], args: Args) -> None:
    """Save experiment results to JSON file.

    Args:
        results: Dictionary containing experiment results
        args: Configuration arguments
    """
    print("Saving results...")
    results["args"] = asdict(args)

    save_dir = Path(f"results/{args.dataset}/cyclicN")
    save_dir.mkdir(parents=True, exist_ok=True)

    json_filename = f"cyclic_alignment_results_cls{args.model}_seed{args.seed}.json"
    save_path = save_dir / json_filename

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")


def main() -> None:
    """Main function to run the rotation experiments."""
    args = tyro.cli(Args)
    print(f"Arguments: {args}")

    # Setup environment
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset and prompts
    dataset = get_target_dataset(args.dataset, transform=transforms.ToTensor())
    prompts = load_prompts(args.dataset)

    # Initialize classifier
    classifier_map = {
        "clip": CLIPClassifier,
        "resnet": ResNet50,
        "vitb": ViTB,
        "dino": DINOv2Classifier,
        "prlc_r50": PRLC_R50,
        "prlc_vit": PRLC_ViTB,
    }
    classifier = classifier_map[args.model](
        prompts, dataset=args.dataset, device=device, ckpt=args.ckpt
    )

    # Validate configuration
    use_diffusion = args.diffusion.factor > 0
    use_clip_energy = args.clip_energy.factor > 0

    if not (use_diffusion or use_clip_energy):
        raise ValueError(
            "Need to use at least one alignment method. Did you mean to set clip_energy.factor or diffusion.factor to a positive value?"
        )
    # Initialize alignment models
    clip_model = None
    if use_clip_energy:
        if args.model != "clip":
            clip_model = CLIPClassifier(prompts, dataset=args.dataset, device=device)
        else:
            clip_model = classifier

    diffusion_model = None
    octagon_mask = None
    if use_diffusion:
        diffusion_model = StableDiffusion(
            model_id=args.diffusion.model, size=args.diffusion.size, device=device
        )
        # Note: Diffusion energy is sensitive to the padding at the corners from non-90 degree rotations.
        # To mitigate this, we use an octagon mask to mask out the corners of the rotated image.
        octagon_mask = torch.ones((1, 1, 224, 224), dtype=torch.float32, device=device)
        octagon_mask = torch_rotate(
            octagon_mask, 45, interpolation=InterpolationMode.NEAREST
        )

    # Setup experiment parameters
    num_samples = len(dataset) if args.N == -1 else args.N
    angles = np.linspace(-180, 180, args.num_angles, endpoint=False)
    print(f"Angles: {angles}")

    # Initialize results dictionary
    results = {
        "correct": [],
        "correct_after_rot": [],
        "correct_after_rot_and_unrot": [],
        "correct_after_rot_and_realign": [],
        "correct_upright_align": [],
        "thetas": [],
        "pose_accuracy": [],
        "pose_dist": [],
        "inferred_angles": [],
        "labels": [],
    }

    # Run experiments
    with torch.inference_mode():
        pbar = tqdm(range(num_samples), dynamic_ncols=True)
        for idx in pbar:
            theta = np.random.choice(angles)
            results["thetas"].append(theta)

            im, label = dataset[idx]
            im = resize_crop(im).to(device).unsqueeze(0)

            results["labels"].append(int(label))

            # Default accuracy
            results["correct"].append(evaluate_img(im, classifier, label))

            # Rotate image
            im_rot = rotate(im.clone(), theta)
            results["correct_after_rot"].append(evaluate_img(im_rot, classifier, label))

            # Oracle accuracy (rotate back)
            im_unrot = rotate(im_rot.clone(), -theta)
            results["correct_after_rot_and_unrot"].append(
                evaluate_img(im_unrot, classifier, label)
            )

            if use_diffusion or use_clip_energy:
                rot_ims = torch.cat(
                    [rotate(im_rot.clone().squeeze().unsqueeze(0), a) for a in angles]
                )

                diff_score = 0
                if use_diffusion and diffusion_model is not None:
                    masked_ims = rot_ims
                    if octagon_mask is not None:
                        masked_ims = rot_ims * octagon_mask
                    diff_score = diff_energy(
                        masked_ims, diffusion_model, args.diffusion
                    )

                uncond_cls_score = 0
                if use_clip_energy and clip_model is not None:
                    uncond_cls_score = uncond_clip_energy(
                        rot_ims, clip_model, args.clip_energy
                    )

                final_score = (
                    args.diffusion.factor * diff_score
                    + args.clip_energy.factor * uncond_cls_score
                )
                best_angle = angles[np.argmin(final_score)]
            else:
                raise ValueError("No alignment method specified")

            im_realign = rotate(im_rot.clone(), best_angle)
            results["correct_after_rot_and_realign"].append(
                evaluate_img(im_realign, classifier, label)
            )
            results["inferred_angles"].append(best_angle)

            # Calculate pose accuracy and distance
            # The ideal realignment angle is -theta, but wrapped around the circle.
            # The model predicts `best_angle`. We remap the predicted angle.
            best_angle_remapped = ((180 - best_angle) % 360) - 180
            results["pose_accuracy"].append(
                int(abs(theta - best_angle_remapped) < 1e-6)
            )

            theta_cossin = np.array(
                [np.cos(theta * np.pi / 180), np.sin(theta * np.pi / 180)]
            )
            pred_cossin = np.array(
                [
                    np.cos(best_angle_remapped * np.pi / 180),
                    np.sin(best_angle_remapped * np.pi / 180),
                ]
            )
            cosine_similarity = (theta_cossin * pred_cossin).sum()
            cosine_similarity = np.clip(cosine_similarity, -1, 1)
            results["pose_dist"].append(np.rad2deg(np.arccos(cosine_similarity)))

            # Process upright alignment
            if use_diffusion or use_clip_energy:
                rot_ims = torch.cat(
                    [rotate(im.clone().squeeze().unsqueeze(0), a) for a in angles]
                )

                diff_score = 0
                if use_diffusion and diffusion_model is not None:
                    masked_ims = rot_ims
                    if octagon_mask is not None:
                        masked_ims = rot_ims * octagon_mask
                    diff_score = diff_energy(
                        masked_ims, diffusion_model, args.diffusion
                    )

                uncond_cls_score = 0
                if use_clip_energy and clip_model is not None:
                    uncond_cls_score = uncond_clip_energy(
                        rot_ims, clip_model, args.clip_energy
                    )

                final_score = (
                    args.diffusion.factor * diff_score
                    + args.clip_energy.factor * uncond_cls_score
                )
                best_angle = angles[np.argmin(final_score)]
            else:
                raise ValueError("No alignment method specified")

            im_realign = rotate(im.clone(), best_angle)
            results["correct_upright_align"].append(
                evaluate_img(im_realign, classifier, label)
            )

            # Update progress bar
            pbar.set_postfix_str(generate_stats(results))

    # Print and save final results
    print(generate_stats(results))
    save_results(results, args)


if __name__ == "__main__":
    main()
