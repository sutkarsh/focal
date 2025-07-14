"""
daynight.py
Runs day/night image transformation experiments using latent space interpolation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import tyro

from focal.utils.datasets import DayNightDataset, load_prompts
from focal.utils.classifiers import CLIPClassifier
from focal.utils.diffusion import StableDiffusion
from focal.utils.energy import CLIPEnergyArgs, uncond_clip_energy


# Stable Diffusion VAE scaling factor
VAE_SCALE_FACTOR = 0.18215


@dataclass(frozen=True)
class Args:
    """Configuration for Day/Night transformation experiments.

    This class defines parameters for running experiments where night images
    are transformed towards day images in the latent space.
    """

    dataset_dir: str = "/home/utkarsh/urbanir/results_valid"
    """Directory containing the day/night image pairs"""

    diffusion_size: int = 512
    """Size of images for stable diffusion model"""

    diffusion_device: str = "cuda:0"
    """Device to run the diffusion model on"""

    num_samples: int = 4
    """Number of image pairs to analyze"""

    num_interpolation_steps: int = 80
    """Number of steps for interpolation between night and day"""

    interpolation_range: Tuple[float, float] = (0, 1.0)
    """Range for interpolation coefficient alpha"""

    classifier_device: str = "cuda:1"
    """Device to run the CLIP classifier on"""

    results_dir: str = "results"
    """Directory to save results"""


def extract_latents(
    dataset: DayNightDataset, diffusion_model: StableDiffusion, num_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract latent representations for day and night image pairs.

    Args:
        dataset: Dataset containing day/night image pairs
        diffusion_model: Stable diffusion model for encoding images
        num_samples: Number of image pairs to process

    Returns:
        Tuple of (day_latents, night_latents) as numpy arrays
    """
    day_latents = []
    night_latents = []

    for idx in tqdm(range(num_samples), desc="Extracting latents"):
        day, night = dataset[idx]
        with torch.inference_mode():
            # Transform and encode day image
            day_batch = diffusion_model.transform(
                day.unsqueeze(0).half().to(diffusion_model.device)
            )
            day_latent = (
                diffusion_model.pipe.vae.encode(day_batch)
                .latent_dist.mean.cpu()
                .detach()
                .numpy()
                * VAE_SCALE_FACTOR
            )

            # Transform and encode night image
            night_batch = diffusion_model.transform(
                night.unsqueeze(0).half().to(diffusion_model.device)
            )
            night_latent = (
                diffusion_model.pipe.vae.encode(night_batch)
                .latent_dist.mean.cpu()
                .detach()
                .numpy()
                * VAE_SCALE_FACTOR
            )

            day_latents.append(day_latent)
            night_latents.append(night_latent)

    return np.concatenate(day_latents), np.concatenate(night_latents)


def get_interpolated_image(
    night_latent: np.ndarray,
    day_latent: np.ndarray,
    diffusion_model: StableDiffusion,
    alpha: float = 0,
) -> torch.Tensor:
    """Generate interpolated image from night and day latents.

    Args:
        night_latent: Latent representation of night image
        day_latent: Latent representation of day image
        diffusion_model: Stable diffusion model for decoding
        alpha: Interpolation coefficient (0 = night, 1 = day)

    Returns:
        Interpolated image as a tensor
    """
    with torch.inference_mode():
        # Interpolate in latent space
        new_latent = torch.tensor(
            alpha * day_latent + (1 - alpha) * night_latent,
            device=diffusion_model.device,
        )

        # Decode to image
        dec_interpolated = diffusion_model.pipe.decode_latents(
            new_latent.half().unsqueeze(0)
        )

    return dec_interpolated[0]


def analyze_interpolation_curves(
    night_latents: np.ndarray,
    day_latents: np.ndarray,
    diffusion_model: StableDiffusion,
    classifier: CLIPClassifier,
    num_samples: int,
    num_steps: int,
    alpha_range: Tuple[float, float],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Analyze classifier energy curves for day/night interpolation.

    Args:
        night_latents: Latent representations of night images
        day_latents: Latent representations of day images
        diffusion_model: Stable diffusion model
        classifier: CLIP classifier for energy computation
        num_samples: Number of image pairs to analyze
        num_steps: Number of interpolation steps
        alpha_range: Range for interpolation coefficient

    Returns:
        Tuple of (alpha values, energy curves)
    """
    curves = []
    alphas = np.linspace(alpha_range[0], alpha_range[1], num_steps)

    for idx in tqdm(range(num_samples), desc="Analyzing curves"):
        night_latent = night_latents[idx]
        day_latent = day_latents[idx]

        # Define energy function for current image pair
        def compute_energy(alpha: float) -> float:
            img = get_interpolated_image(
                night_latent, day_latent, diffusion_model, alpha=alpha
            )
            img_tensor = torch.tensor(img).permute(2, 0, 1)[None]
            return uncond_clip_energy(
                img_tensor,
                classifier,
                args=CLIPEnergyArgs(factor=1, logit_top_factor=-2),
            ).item()

        # Compute energy curve
        curve = np.array([compute_energy(alpha) for alpha in alphas])
        curves.append(curve)

    return alphas, curves


def plot_and_save_curves(
    alphas: np.ndarray, curves: List[np.ndarray], save_path: str
) -> None:
    """Plot and save interpolation energy curves.

    Args:
        alphas: Interpolation coefficient values
        curves: List of energy curves for each image pair
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    for curve in curves:
        plt.plot(alphas, -curve)

    plt.xlabel("Interpolation Coefficient (α)")
    plt.ylabel("Negative Energy")
    plt.title("Day/Night Transformation Energy Curves")
    plt.grid(True)

    # Create directory if needed
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    """Main function to run the day/night transformation experiments."""
    args = tyro.cli(Args)
    print(f"Arguments: {args}")

    # Initialize models and dataset
    dataset = DayNightDataset(dataset_dir=args.dataset_dir)
    diffusion_model = StableDiffusion(
        size=args.diffusion_size, device=args.diffusion_device
    )
    classifier = CLIPClassifier(
        load_prompts("daynight"), dataset="daynight", device=args.classifier_device
    )

    # Extract latent representations
    day_latents, night_latents = extract_latents(
        dataset, diffusion_model, args.num_samples
    )

    # Analyze interpolation curves
    alphas, curves = analyze_interpolation_curves(
        night_latents,
        day_latents,
        diffusion_model,
        classifier,
        args.num_samples,
        args.num_interpolation_steps,
        args.interpolation_range,
    )

    # Plot and save results
    save_path = Path(args.results_dir) / "daynight.png"
    plot_and_save_curves(alphas, curves, str(save_path))


if __name__ == "__main__":
    main()
