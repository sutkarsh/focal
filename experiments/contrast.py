"""
contrast.py
Runs gamma transform (contrast) experiments.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import tyro
from dataclasses import dataclass, asdict
from bayes_opt import UtilityFunction, BayesianOptimization
from sklearn.gaussian_process.kernels import RBF

from focal.utils.datasets import get_target_dataset, load_prompts
from focal.utils.energy import CLIPEnergyArgs, DiffusionEnergyArgs, uncond_clip_energy
from focal.utils.classifiers import (
    CLIPClassifier,
    ResNet50,
    ViTB,
    DINOv2Classifier,
)


@dataclass(frozen=True)
class BayesianOptArgs:
    """Arguments for Bayesian optimization"""

    init_grid_resolution: int = 3
    """Number of gamma values for transformation"""

    init_random_points: int = 4
    """Number of initial points for BO"""

    iter_points: int = 5
    """Number of iteration points for BO"""

    opt_range: float = 2
    """Optimization range in gamma space"""


@dataclass(frozen=True)
class Args:
    """Configuration for Contrast/Gamma Alignment experiments."""

    dataset: Literal["cifar10", "cifar100", "stl10", "imagenet"] = "imagenet"
    """Dataset to use for experiments"""

    model: Literal["clip", "resnet", "vitb", "dino"] = "clip"
    """Model architecture for downstream classification"""

    N_min: int = -1
    """Start index for processing (-1 for beginning)"""

    N_max: int = -1
    """End index for processing (-1 for end)"""

    gamma_range: float = 2
    """Gamma range in log space"""

    diffusion: DiffusionEnergyArgs = DiffusionEnergyArgs()
    """Diffusion energy arguments"""

    clip_energy: CLIPEnergyArgs = CLIPEnergyArgs(factor=1)
    """Classifier energy arguments"""

    bo: BayesianOptArgs = BayesianOptArgs()
    """Bayesian optimization arguments"""

    seed: int = 0
    """Random seed for reproducibility"""


def apply_contrast(im: torch.Tensor, gamma_log: float) -> torch.Tensor:
    """Apply gamma/contrast transformation to an image.

    Args:
        im: Input image tensor (N, 3, H, W)
        gamma_log: Gamma value in log space (will be exponentiated)

    Returns:
        Transformed image tensor with gamma correction applied (N, 3, H, W)
    """
    assert im.dim() == 4, "Input image must be a 4D tensor (N, 3, H, W)"
    assert im.size(1) == 3, (
        f"Input image must have 3 channels (RGB), got shape {im.shape}"
    )

    # Normalize to [0,1] range
    im = im / im.max()
    im = im.clamp(0, 1)

    # Convert from log space to actual gamma value
    gamma = np.exp(gamma_log)

    # Apply gamma transformation
    im = im.pow(gamma)

    # Normalize
    im = im / im.max().item()
    return im


def evaluate_img(
    img: torch.Tensor,
    classifier: Union[CLIPClassifier, ResNet50, ViTB, DINOv2Classifier],
    label: int,
) -> int:
    """Evaluate if classifier prediction matches the label.

    Args:
        img: Input image tensor (1, 3, H, W)
        classifier: Classifier model
        label: Ground truth label

    Returns:
        1 if prediction is correct, 0 otherwise
    """
    assert img.dim() == 4, "Input image must be a 4D tensor (N, 3, H, W)"
    assert img.size(1) == 3, (
        f"Input image must have 3 channels (RGB), got shape {img.shape}"
    )

    pred = classifier(img / img.max().item())
    pred_label = pred.argmax().cpu().item()
    return int(label == pred_label)


def run_BO(
    target_fn: Callable[[float], float],
    init_gamma_grid: np.ndarray = None,
    init_random_points: int = 10,
    n_iter: int = 10,
    opt_range: Tuple[float, float] = (-1.5, 0),
) -> BayesianOptimization:
    """Run Bayesian optimization for gamma/contrast alignment.

    Args:
        target_fn: Function to optimize
        init_gamma_grid: Initial gamma grid to probe (in log space)
        init_random_points: Number of initial points sampled randomly
        n_iter: Number of iterations
        opt_range: Range for optimization in log space

    Returns:
        Optimized BayesianOptimization object
    """
    util_func = UtilityFunction(kind="ei", xi=0.0)
    optimizer = BayesianOptimization(
        target_fn,
        {"u1": opt_range},
        random_state=27,
        verbose=0,
        allow_duplicate_points=True,
    )

    optimizer.set_gp_params(kernel=RBF(length_scale=0.3))

    if init_gamma_grid is not None:
        for gamma in init_gamma_grid:
            optimizer.probe(params={"u1": gamma})

    optimizer.maximize(
        init_points=init_random_points,
        n_iter=n_iter,
        acquisition_function=util_func,
    )

    return optimizer


def generate_stats(results: Dict[str, List[Any]]) -> str:
    """Generate statistics string from results.

    Args:
        results: Dictionary containing experiment results

    Returns:
        Formatted string of statistics
    """
    stats = [
        f"Default: {np.mean(results['correct']) * 100:.3f}%",
        f"Gamma: {np.mean(results['correct_after_gamma']) * 100:.3f}%",
        f"Gamma+Ungamma: {np.mean(results['correct_after_gamma_and_ungamma']) * 100:.3f}%",
        f"Gamma+Realign: {np.mean(results['correct_after_gamma_and_realign']) * 100:.3f}%",
        f"Gamma L2 (med): {np.median(results['gamma_accuracy']):.3f}",
        f"L2 (mean): {np.mean(results['gamma_accuracy']):.3f}",
    ]
    return " ".join(stats)


def save_results(results: Dict[str, Any], args: Args, N_min: int, N_max: int) -> None:
    """Save experiment results to a JSON file.

    Args:
        results: Dictionary containing experiment results.
        args: Configuration arguments.
        N_min: Start index of processed samples.
        N_max: End index of processed samples.
    """
    print("Saving results...")
    results["args"] = asdict(args)

    save_dir = Path(f"results/{args.dataset}_gamma_BO_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    json_filename = (
        f"gamma_BO_results_n{N_min}_n{N_max}_seed{args.seed}_model{args.model}.json"
    )
    save_path = save_dir / json_filename

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")


def main() -> None:
    """Main function to run the gamma/contrast experiments."""
    args = tyro.cli(Args)
    print(f"Arguments: {args}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and prompts
    dataset = get_target_dataset(
        args.dataset, train=False, transform=transforms.ToTensor()
    )
    prompts = load_prompts(args.dataset)

    # Initialize classifier
    classifier_map = {
        "clip": CLIPClassifier,
        "resnet": ResNet50,
        "vitb": ViTB,
        "dino": DINOv2Classifier,
    }
    classifier = classifier_map[args.model](
        prompts, dataset=args.dataset, device=device
    )

    # Validate configuration
    use_clip_energy = args.clip_energy.factor > 0
    assert use_clip_energy, "Need to set clip_energy.factor > 0 for alignment"

    # Initialize alignment models
    clip_model = None
    if use_clip_energy:
        if args.model != "clip":
            clip_model = CLIPClassifier(prompts, dataset=args.dataset, device=device)
        else:
            clip_model = classifier

    # Setup experiment parameters
    N_min, N_max = (0, len(dataset)) if args.N_min == -1 else (args.N_min, args.N_max)
    init_gamma_grid = np.linspace(
        -args.gamma_range, args.gamma_range, args.bo.init_grid_resolution, endpoint=True
    )

    # Initialize results dictionary
    results: Dict[str, List[Any]] = {
        "correct": [],
        "correct_after_gamma": [],
        "correct_after_gamma_and_ungamma": [],
        "correct_after_gamma_and_realign": [],
        "gammas": [],
        "gamma_accuracy": [],
        "predicted_gammas": [],
    }

    # Run experiments
    with torch.inference_mode():
        pbar = tqdm(range(N_min, N_max), dynamic_ncols=True)
        for idx in pbar:
            gamma = np.random.uniform(-args.gamma_range, args.gamma_range)
            results["gammas"].append(gamma)

            im, label = dataset[idx]
            im = im.to(device).unsqueeze(0)

            # 1. Default accuracy
            results["correct"].append(evaluate_img(im, classifier, label))

            # 2. Accuracy after applying a random gamma transform
            im_gamma = apply_contrast(im.clone(), gamma)
            results["correct_after_gamma"].append(
                evaluate_img(im_gamma, classifier, label)
            )

            # 3. Oracle accuracy (reversing the known gamma transform)
            # Applying a negative gamma in log space is equivalent to reciprocal gamma (1/gamma).
            im_ungamma = apply_contrast(im_gamma.clone(), -gamma)
            results["correct_after_gamma_and_ungamma"].append(
                evaluate_img(im_ungamma, classifier, label)
            )

            # Define the target function for Bayesian Optimization
            def bo_target_fn(u1: float) -> float:
                """
                Calculates the alignment score for a given gamma guess (u1).
                The goal is to find the u1 that maximizes this score.
                """
                with torch.inference_mode():
                    im_base = im_gamma.clone().squeeze().unsqueeze(0)
                    im_gamma_guess = apply_contrast(im_base, u1)

                    final_score = 0.0
                    if use_clip_energy:
                        # Negate CLIP energy because BO maximizes, but we want to minimize energy
                        cls_score = -uncond_clip_energy(
                            im_gamma_guess, clip_model, args.clip_energy
                        )
                        final_score += args.clip_energy.factor * cls_score.item()

                    return final_score

            # Run Bayesian optimization
            optimizer = run_BO(
                bo_target_fn,
                init_random_points=args.bo.init_random_points,
                n_iter=args.bo.iter_points,
                opt_range=(-args.bo.opt_range, args.bo.opt_range),
                init_gamma_grid=init_gamma_grid,
            )
            best_gamma = optimizer.max["params"]["u1"]
            results["predicted_gammas"].append(best_gamma)

            # 4. Accuracy after realignment using the best gamma found by BO
            im_realign = apply_contrast(im_gamma.clone(), best_gamma)
            results["correct_after_gamma_and_realign"].append(
                evaluate_img(im_realign, classifier, label)
            )

            # The ideal realignment `best_gamma` should be the inverse of the original `gamma`.
            # We measure the absolute difference between `best_gamma` and `-gamma`.
            results["gamma_accuracy"].append(abs(gamma + best_gamma))

    # Print and save final results
    print(generate_stats(results))
    save_results(results, args, N_min, N_max)


if __name__ == "__main__":
    main()
