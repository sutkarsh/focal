"""
chrom.py
Runs chrominance transformation experiments.
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
from focal.utils.classifiers import CLIPClassifier, ResNet50, ViTB, DINOv2Classifier
from focal.utils.energy import DiffusionEnergyArgs, CLIPEnergyArgs, uncond_clip_energy


@dataclass(frozen=True)
class BayesianOptArgs:
    """Arguments for Bayesian optimization"""

    init_grid_resolution: int = 3
    """Number of chrominance values for transformation"""

    init_random_points: int = 6
    """Number of initial points for BO"""

    iter_points: int = 20
    """Number of iteration points for BO"""


@dataclass(frozen=True)
class Args:
    """Configuration for Chrominance Alignment experiments."""

    dataset: Literal["cifar10", "cifar100", "stl10", "imagenet"] = "cifar100"
    """Dataset to use for experiments"""

    model: Literal["clip", "resnet", "vitb", "dino"] = "clip"
    """Model architecture for downstream classification"""

    N_min: int = -1
    """Start index for processing (-1 for beginning)"""

    N_max: int = -1
    """End index for processing (-1 for end)"""

    chrom_range: float = 1
    """Chrominance range in log space"""

    diffusion: DiffusionEnergyArgs = DiffusionEnergyArgs()
    """Diffusion energy arguments"""

    clip_energy: CLIPEnergyArgs = CLIPEnergyArgs(factor=1)
    """Classifier energy arguments"""

    bo: BayesianOptArgs = BayesianOptArgs()
    """Bayesian optimization arguments"""

    seed: int = 0
    """Random seed for reproducibility"""


def apply_chrom(im: torch.Tensor, chrom: torch.Tensor) -> torch.Tensor:
    """Apply chrominance transformation to an image.

    Args:
        im: Input image tensor (N, 3, H, W)
        chrom: Chrominance values tensor (2,)

    Returns:
        Transformed image tensor
    """
    assert im.dim() == 4, "Input image must be a 4D tensor (N, C, H, W)"
    assert im.size(1) == 3, (
        f"Input image must have 3 channels (RGB), got shape {im.shape}"
    )
    assert chrom.dim() == 1 and chrom.size(0) == 2, (
        f"Chrominance must be a 1D tensor of size 2, got shape {chrom.shape}"
    )

    L_rgb = torch.tensor(
        [torch.exp(-chrom[0]), 1, torch.exp(-chrom[1])], device=im.device
    ).float()
    L_rgb = L_rgb / L_rgb.norm()
    im = im * L_rgb[:, None, None]
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
    assert img.dim() == 4 and img.size(0) == 1, (
        f"Input image must be a 4D tensor with batch size 1, got shape {img.shape}"
    )
    assert img.size(1) == 3, (
        f"Input image must have 3 channels (RGB), got shape {img.shape}"
    )
    pred = classifier(img / img.max().item())
    pred_label = pred.argmax().cpu().item()
    return int(label == pred_label)


def run_BO(
    target_fn: Callable[[float, float], float],
    init_chrom_grid: np.ndarray = None,
    init_random_points: int = 10,
    n_iter: int = 10,
    opt_range: Tuple[float, float] = (-1, 1),
) -> BayesianOptimization:
    """Run Bayesian optimization for chrominance alignment.

    Args:
        target_fn: Function to optimize
        init_chrom_grid: Initial chrominance grid to probe
        init_random_points: Number of initial points sampled randomly
        n_iter: Number of iterations
        opt_range: Range for optimization

    Returns:
        Optimized BayesianOptimization object
    """
    # Set up Bayesian optimization solver
    util_func = UtilityFunction(kind="ei", xi=0.0)
    optimizer = BayesianOptimization(
        target_fn,
        {"u1": opt_range, "u2": opt_range},
        random_state=27,
        verbose=0,
        allow_duplicate_points=True,
    )
    optimizer.set_gp_params(kernel=RBF(length_scale=0.3))

    # Probe initial points before running BO
    if init_chrom_grid is not None:
        for chrom in init_chrom_grid:
            optimizer.probe(params={"u1": chrom[0], "u2": chrom[1]})

    # Run BO on the energy function
    optimizer.maximize(
        init_points=init_random_points, n_iter=n_iter, acquisition_function=util_func
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
        f"Chrom: {np.mean(results['correct_after_chrom']) * 100:.3f}%",
        f"Chrom+Unchrom: {np.mean(results['correct_after_chrom_and_unchrom']) * 100:.3f}%",
        f"Chrom+Realign: {np.mean(results['correct_after_chrom_and_realign']) * 100:.3f}%",
        f"Chrom L2 (med): {np.median(results['chrom_accuracy']):.3f}",
        f"L2 (mean): {np.mean(results['chrom_accuracy']):.3f}",
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

    # Use pathlib for robust path handling
    save_dir = Path(f"results/{args.dataset}_chrom_BO_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    json_filename = f"chrom_BO_results_n{N_min}_n{N_max}_seed{args.seed}.json"
    save_path = save_dir / json_filename

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")


def main() -> None:
    """Main function to run the chrominance experiments."""
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
    use_diffusion = args.diffusion.factor > 0
    assert not use_diffusion, "This experiment uses CLIP only"
    use_clip_energy = args.clip_energy.factor > 0
    assert use_diffusion or use_clip_energy, "Need to use at least one alignment method"

    # Initialize alignment models
    clip_model = None
    if use_clip_energy:
        if args.model != "clip":
            clip_model = CLIPClassifier(prompts, dataset=args.dataset).to(device)
        else:
            clip_model = classifier

    # Setup experiment parameters
    N_min, N_max = (0, len(dataset)) if args.N_min == -1 else (args.N_min, args.N_max)

    # Create a grid of initial chrominance values to probe during Bayesian Optimization
    init_chrom_grid = np.array(
        [
            (x, y)
            for x in np.linspace(
                -args.chrom_range,
                args.chrom_range,
                args.bo.init_grid_resolution,
                endpoint=True,
            )
            for y in np.linspace(
                -args.chrom_range,
                args.chrom_range,
                args.bo.init_grid_resolution,
                endpoint=True,
            )
        ]
    )

    # Initialize a dictionary to store results for each metric
    results: Dict[str, List[Any]] = {
        "correct": [],
        "correct_after_chrom": [],
        "correct_after_chrom_and_unchrom": [],
        "correct_after_chrom_and_realign": [],
        "chroms": [],
        "chrom_accuracy": [],
        "predicted_chroms": [],
    }

    # Run experiments
    with torch.inference_mode():
        pbar = tqdm(range(N_min, N_max), dynamic_ncols=True)
        for idx in pbar:
            chrom = np.random.uniform(-args.chrom_range, args.chrom_range, 2)
            results["chroms"].append([chrom[0], chrom[1]])

            im, label = dataset[idx]
            im = im.to(device).unsqueeze(0)

            # 1. Default accuracy on the original image
            results["correct"].append(evaluate_img(im, classifier, label))

            # 2. Accuracy after applying a random chrominance shift
            im_chrom = apply_chrom(im.clone(), torch.tensor(chrom, device=im.device))
            results["correct_after_chrom"].append(
                evaluate_img(im_chrom, classifier, label)
            )

            # 3. Oracle accuracy (reversing the known chrominance shift)
            im_unchrom = apply_chrom(
                im_chrom.clone(), torch.tensor(-chrom, device=im.device)
            )
            results["correct_after_chrom_and_unchrom"].append(
                evaluate_img(im_unchrom, classifier, label)
            )

            # Define the target function for Bayesian Optimization
            def bo_target_fn(u1: float, u2: float) -> float:
                """
                Calculates the alignment score for a given chrominance guess (u1, u2).
                The goal is to find the (u1, u2) that maximizes this score.
                """
                with torch.inference_mode():
                    im_base = im_chrom.clone().squeeze().unsqueeze(0)
                    chrom_tensor = torch.tensor([u1, u2], device=im_base.device)
                    im_chrom_guess = apply_chrom(im_base, chrom_tensor)

                    # Normalize image once before energy calculations
                    im_chrom_guess = im_chrom_guess / im_chrom_guess.max().item()

                    final_score = 0

                    if use_clip_energy:
                        # Negate CLIP energy because Bayesian Optimization maximizes,
                        # while we want to minimize the energy.
                        cls_score = -uncond_clip_energy(
                            im_chrom_guess, clip_model, args.clip_energy
                        )
                        final_score += args.clip_energy.factor * cls_score.item()

                return final_score

            # Run Bayesian optimization
            optimizer = run_BO(
                bo_target_fn,
                init_random_points=args.bo.init_random_points,
                n_iter=args.bo.iter_points,
                opt_range=(-args.chrom_range, args.chrom_range),
                init_chrom_grid=init_chrom_grid,
            )
            best_chrom = np.array(
                [optimizer.max["params"]["u1"], optimizer.max["params"]["u2"]]
            )
            results["predicted_chroms"].append([best_chrom[0], best_chrom[1]])

            # 4. Accuracy after realignment using the best chrominance found by BO
            im_realign = apply_chrom(
                im_chrom.clone(), torch.tensor(best_chrom, device=im.device)
            )
            results["correct_after_chrom_and_realign"].append(
                evaluate_img(im_realign, classifier, label)
            )

            # The ideal realignment `best_chrom` should be the inverse of the original `chrom`.
            # We measure the L2 distance between `best_chrom` and `-chrom`.
            results["chrom_accuracy"].append(np.linalg.norm(best_chrom + chrom))

            # Update progress bar
            pbar.set_postfix_str(generate_stats(results))

    # Print and save final results
    print(generate_stats(results))
    save_results(results, args, N_min, N_max)


if __name__ == "__main__":
    main()
