"""
Active vision experiments using Gaussian splatting scenes with CLIP and diffusion guidance.
"""

import os
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from bayes_opt import BayesianOptimization, UtilityFunction
from scipy.spatial.transform import Rotation
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm

from focal.utils.energy import CLIPEnergyArgs, uncond_clip_energy
from focal.utils.classifiers import CLIPClassifier
from focal.utils.datasets import load_prompts
from focal.utils.nerf_dataset import Camera, NeRFDataset, render


@dataclass(frozen=True)
class ActiveVisionArgs:
    """Configuration for active vision experiments."""

    seed: int = 1
    """Random seed for reproducibility"""

    init_points: int = 450
    """Number of initial points for Bayesian optimization"""

    n_iter: int = 150
    """Number of iteration points for Bayesian optimization"""

    translation_range: float = 3.0
    """Range for translation parameters"""

    rotation_range: float = 1.0
    """Range for rotation parameters"""

    clip_energy: CLIPEnergyArgs = CLIPEnergyArgs(factor=1, logit_top_factor=-0.33)
    """Classifier energy arguments"""

    image_size: int = 512
    """Size to resize rendered images to"""


def set_plotting_style() -> None:
    """Set matplotlib plotting style parameters."""
    params = {
        "legend.fontsize": 22,
        "axes.labelsize": 30,
        "axes.titlesize": 30,
        "xtick.labelsize": 27,
        "ytick.labelsize": 22,
    }
    plt.rcParams.update(params)

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["mathtext.rm"] = "serif"
    matplotlib.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "cmr10"


def render_view(
    dset: NeRFDataset,
    Tx: float = 0,
    Ty: float = 0,
    Tz: float = 0,
    q1: float = 0,
    q2: float = 0,
    q3: float = 0,
) -> torch.Tensor:
    """Render a view of the scene with given camera parameters.

    Args:
        dset: NeRF dataset object
        Tx: Translation in x direction
        Ty: Translation in y direction
        Tz: Translation in z direction
        q1: First quaternion component
        q2: Second quaternion component
        q3: Third quaternion component

    Returns:
        Rendered image tensor
    """
    assert dset.currently_loaded_scene is not None, (
        "No scene loaded. Call _load_scene first"
    )

    # Set up translation
    newT = np.array([Tx, Ty, Tz])

    # Set up rotation using quaternion
    quat = torch.tensor([q1, q2, q3, 0.0])
    quat[-1] = torch.sqrt(1.0 - torch.norm(quat) ** 2)
    quat += 1e-6 * torch.randn(4)  # Add noise to avoid singularity
    quat = torch.nan_to_num(quat, posinf=0.0, neginf=0.0)
    quat /= torch.norm(quat)

    newR = Rotation.from_quat(quat).as_matrix()

    # Create new camera view
    newView = Camera(
        colmap_id=dset.view.uid,
        R=newR,
        T=newT,
        FoVx=dset.view.FoVx,
        FoVy=dset.view.FoVy,
        image=dset.view.original_image,
        gt_alpha_mask=None,
        image_name=dset.view.image_name,
        uid=dset.view.uid,
        data_device=dset.view.data_device,
    )

    with torch.no_grad():
        rendering = render(newView, dset.gaussians, dset.pipeline, dset.background)[
            "render"
        ]
    return rendering


def get_render(
    dset: NeRFDataset,
    args: ActiveVisionArgs,
    Tx: float = 0,
    Ty: float = 0,
    Tz: float = 0,
    q1: float = 0,
    q2: float = 0,
    q3: float = 0,
) -> torch.Tensor:
    """Get processed render of the scene.

    Args:
        dset: NeRF dataset object
        args: Configuration arguments
        Tx: Translation in x direction
        Ty: Translation in y direction
        Tz: Translation in z direction
        q1: First quaternion component
        q2: Second quaternion component
        q3: Third quaternion component

    Returns:
        Processed image tensor
    """
    im = render_view(dset, Tx=Tx, Ty=Ty, Tz=Tz, q1=q1, q2=q2, q3=q3).cpu()

    # Pad image to form a square
    _, h, w = im.shape
    max_dim = max(w, h)
    pad_w, pad_h = (max_dim - w) // 2, (max_dim - h) // 2
    padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)

    im = transforms.Pad(padding)(im)
    im = transforms.Resize(args.image_size)(im)
    return im


def optimize_and_visualize(
    dset: NeRFDataset, args: ActiveVisionArgs, clip: CLIPClassifier
) -> None:
    """Run Bayesian optimization and visualize results for a single scene."""

    # Define optimization target function
    def render_and_clip_energy(Tx=0, Ty=0, Tz=0, q1=0, q2=0, q3=0) -> float:
        im = get_render(dset, args, Tx=Tx, Ty=Ty, Tz=Tz, q1=q1, q2=q2, q3=q3)
        return -uncond_clip_energy(im[None], clip, args.clip_energy).item()

    # Setup Bayesian optimization
    util_func = UtilityFunction(kind="ei", xi=0)
    optimizer = BayesianOptimization(
        f=render_and_clip_energy,
        pbounds={
            "Tx": (-args.translation_range, args.translation_range),
            "Ty": (-args.translation_range, args.translation_range),
            "Tz": (-args.translation_range, args.translation_range),
            "q1": (-args.rotation_range, args.rotation_range),
            "q2": (-args.rotation_range, args.rotation_range),
            "q3": (-args.rotation_range, args.rotation_range),
        },
        random_state=args.seed,
        verbose=0,
    )
    optimizer.set_gp_params(alpha=0.01, kernel=RBF(length_scale=0.4))

    optimizer.probe(
        params={"Tx": 0, "Ty": 0, "Tz": 0, "q1": 0, "q2": 0, "q3": 0}, lazy=False
    )
    optimizer.maximize(
        init_points=args.init_points,
        n_iter=args.n_iter,
        acquisition_function=util_func,
    )

    # Visualize results
    plt.figure(figsize=(18, 6))
    plt.suptitle(f"Example: {dset.currently_loaded_scene}", fontsize=30)

    # Initial view
    plt.subplot(131)
    plt.imshow(get_render(dset, args).permute(1, 2, 0))
    plt.title("Initialization")
    plt.axis("off")

    # Random view
    plt.subplot(132)
    rand_params = {
        "Tx": np.random.uniform(-args.translation_range, args.translation_range),
        "Ty": np.random.uniform(-args.translation_range, args.translation_range),
        "Tz": np.random.uniform(-args.translation_range, args.translation_range),
        "q1": np.random.uniform(-args.rotation_range, args.rotation_range),
        "q2": np.random.uniform(-args.rotation_range, args.rotation_range),
        "q3": np.random.uniform(-args.rotation_range, args.rotation_range),
    }
    rand_im = get_render(dset, args, **rand_params).permute(1, 2, 0).numpy()
    plt.imshow(rand_im)
    plt.title("Random Parameters")
    plt.axis("off")

    # Optimized view
    plt.subplot(133)
    plt.imshow(get_render(dset, args, **optimizer.max["params"]).permute(1, 2, 0))
    plt.title("Optimized")
    plt.axis("off")

    plt.savefig(
        f"results/active_vision/{dset.currently_loaded_scene}{args.seed}.png",
        bbox_inches="tight",
        pad_inches=0,
    )


def main() -> None:
    """Main function to run active vision experiments."""
    args = ActiveVisionArgs()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Set plotting style
    set_plotting_style()

    # Initialize models
    with torch.no_grad():
        prompts = load_prompts("imagenet")
        clip = CLIPClassifier(prompts, dataset="imagenet", device="cuda:1")

    print("Successfully loaded all models!")

    # Load dataset
    dset = NeRFDataset("./gaussian_splatting/", device="cuda:0")
    print(
        "Running Active Vision (unguided) in the Gaussian splatting scenes:",
        dset.scenes,
    )

    # Create output directories
    os.makedirs("results/active_vision/", exist_ok=True)

    # Process each scene
    for scene in tqdm(dset.scenes):
        print("Current scene:", scene)
        dset._load_scene(scene, root=dset.root)
        print("Successfully loaded scene!")
        optimize_and_visualize(dset, args, clip)


if __name__ == "__main__":
    main()
