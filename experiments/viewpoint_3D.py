import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for deterministic behavior
import torch

torch.use_deterministic_algorithms(True)
import torchvision
import random
from PIL import Image
from typing import List, Optional, Dict

import tyro
from dataclasses import dataclass, field
from pytorch_lightning import seed_everything
import numpy as np
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf, DictConfig
from focal.utils.classifiers import OVSEGClassifier, CLIPClassifier
from focal.utils.trellis import (
    gen_3D_trellis,
    render_video,
)
from focal.utils.energy import (
    CLIPEnergyNormPromptArgs,
    DiffusionEnergyArgs,
    diff_energy,
    uncond_clip_energy_norm_prompt,
)
from focal.utils.diffusion import StableDiffusion
from focal.utils.datasets import (
    load_prompts,
    load_classes,
    ObjaverseDataset,
    CO3DDataset,
    rotate_image_objaverse,
    rotate_image_co3d,
)


@dataclass(frozen=True)
class OVSEGArgs:
    """Configuration for OVSEG classifier experiments."""

    config_file: str = "third_party_modified/ovseg/configs/ovseg_swinB_vitL_demo.yaml"
    """path to config file"""

    opts: list = field(
        default_factory=lambda: [
            "MODEL.WEIGHTS",
            "third_party_modified/ovseg/ovseg_swinbase_vitL14_ft_mpt.pth",
        ]
    )
    """OVSEG options. MODEL.WEIGHTS is the path to the model weights"""


@dataclass(frozen=True)
class TrellisArgs:
    """Configuration for Trellis experiments."""

    indices_to_gen: list = field(default_factory=lambda: [0, 12, 24, 35])
    """Indices of the viewpoints to generate Trellis images for"""

    el_min: int = 30
    """Minimum elevation offset to generate"""

    el_max: int = 31
    """Maximum elevation offset to generate"""

    el_step: int = 30
    """Step size for elevation offset"""

    az_min: int = -180
    """Minimum azimuth offset to generate"""

    az_max: int = 180
    """Maximum azimuth offset to generate"""

    az_step: int = 30
    """Step size for azimuth offset"""


@dataclass(frozen=True)
class Args:
    """Configuration for 3d viewpoint Alignment Classifier experiments.

    This class defines parameters for running viewpoint experiments where images
    are rendered at various viewpoints and then aligned back using various methods.
    """

    seed: int = 42
    """Random seed for reproducibility"""

    ovseg: OVSEGArgs = OVSEGArgs()
    """OVSEG arguments"""

    trellis: TrellisArgs = TrellisArgs()
    """Trellis arguments"""

    canon_2d_pattern: str = "0"  # 0 if rank mode, 5
    """Canon 2D pattern: "0" or 5". Denotes how many 45 degree pre-tilts to use."""

    use_original: bool = True
    """Whether to include the original image and pre-tilted images in the FoCal set"""

    mode: str = "rank"
    """Whether to aggregate results by 'rank' or 'gt_prob'"""

    diffusion: DiffusionEnergyArgs = DiffusionEnergyArgs(factor=5)
    """Diffusion energy arguments"""

    cls_energy: CLIPEnergyNormPromptArgs = CLIPEnergyNormPromptArgs()
    """Classifier energy arguments"""

    placeholdertoken: str = (
        "a crisp digital photograph of a [classname] on a bright white backdrop."
    )
    """Placeholder token for the CLIP text prompt"""

    random_ind_file: str = "datasets/objaverse/random_indices.npy"
    """File specifying random indices for evaluation in 'gt_prob' mode"""

    random_angle_file: str = "datasets/objaverse/random_angles.npy"
    """File specifying random input rotation angles for evaluation in 'gt_prob' mode"""

    dataset: str = "objaverse"
    """Dataset - objaverse or co3d"""

    co3d_thresh: float = 0.3
    """Threshold for CO3D dataset filtering"""

    dataset_cfg_file: str = "confs/dataset_3D.yaml"
    """Path to dataset configuration file"""


def get_gen_grid(
    el_min: int = -60,
    el_max: int = 60,
    el_step: int = 30,
    az_min: int = -180,
    az_max: int = 180,
    az_step: int = 30,
) -> torch.tensor:
    """Return a grid of locations to use for sampling Trellis views

    Args:
        el_min: start elevation
        el_max: end elevation
        el_step: elevation interval
        az_min: start azimuth
        az_max: end azimuth
        az_step: azimuth interval

    Returns:
        A 2d pose tensor containing the el and az of each grid position
    """
    az_gen_grid = np.arange(az_min, az_max, az_step)
    el_gen_grid = np.arange(el_min, el_max, el_step)
    gen_pose_tensor = torch.tensor(
        [[el, az] for el in el_gen_grid for az in az_gen_grid]
    )
    return gen_pose_tensor


def classify_and_compute_energy(
    args, img_list, classes, ovseg_classifier, clip_classifier, diff_model
):
    """Classify a list of images and compute their energies.

    Args:
        args: args object containing the script arguments
        img_list: list of PIL images to classify
        classes: list of classes for classification
        ovseg_classifier: OVSEG classifier instance
        clip_classifier: CLIP classifier instance
        diff_model: StableDiffusion model instance (optional)

    Returns:
        A tuple containing:
        - ovseg_logits: tensor of logits from OVSEG classifier
        - energy: tensor of energy values
    """
    ovseg_logits = torch.stack(
        [ovseg_classifier(img, classes).cpu() for img in img_list], dim=0
    )  # shape: [num_imgs, num_classes]

    img_list_torch = torch.cat(
        [
            rearrange(torchvision.transforms.ToTensor()(img), "c h w -> 1 c h w")
            for img in img_list
        ],
        dim=0,
    )  # shape: [num_imgs, C, H, W]
    cls_energy = uncond_clip_energy_norm_prompt(
        img_list_torch, clip_classifier, args.cls_energy
    )

    if diff_model is not None:
        diff_score = diff_energy(img_list_torch, diff_model, args.diffusion)
        total_energy = (
            args.cls_energy.factor * cls_energy + args.diffusion.factor * diff_score
        )
    else:
        total_energy = cls_energy

    return ovseg_logits, torch.from_numpy(total_energy)


def select_best(ovseg_preds: torch.tensor, energy: torch.tensor):
    """
    Select the best prediction based on energy.

    Args:
        ovseg_preds: tensor of OVSEG predictions
        energy: tensor of energy values

    Returns:
        A tuple containing:
        - best_preds: the best prediction based on energy
        - best_energy: the energy of the best prediction
        - best_index: the index of the best prediction
    """
    # sort the energy tensor and get the indices
    sorted_energy, sorted_indices = torch.sort(energy, dim=0, descending=False)
    # sort the predictions based on the sorted indices
    sorted_preds = torch.gather(ovseg_preds, dim=0, index=sorted_indices)
    # grab the best pred, energy, index
    best_preds = sorted_preds[0].item()
    best_energy = sorted_energy[0].item()
    best_index = sorted_indices[0].item()
    return best_preds, best_energy, best_index


def get_pretilt_angles(canon_2d_pattern: str, best_angle: int) -> List[int]:
    """
    Get the pre-tilt angles based on the canon 2D pattern and best angle.

    Args:
        canon_2d_pattern: string representing the canon 2D pattern
        best_angle: the best angle from the pre-tilted images
    Returns:
        A list of pre-tilt angles to use for the FoCal set.
    """
    if canon_2d_pattern == "5":
        pre_tilt_angles = [
            (best_angle - 90) % 360,
            (best_angle - 45) % 360,
            best_angle,
            (best_angle + 45) % 360,
            (best_angle + 90) % 360,
        ]
    elif canon_2d_pattern == "0":
        pre_tilt_angles = [0]
    else:
        raise ValueError("Invalid canon 2d pattern")
    return pre_tilt_angles


def generate_trellis_images(args, img, yaws, pitches):
    """
    Generate Trellis images for a given image.

    Args:
        args: args object containing the script arguments
        img: PIL image to generate Trellis images for
        yaws: tensor of yaw angles to generate Trellis images for
        pitches: tensor of pitch angles to generate Trellis images for

    Returns:
        A list of PIL images representing the Trellis views.
    """
    try:
        trellis_outputs = gen_3D_trellis(img, preprocess=(args.dataset == "objaverse"))
        gauss = trellis_outputs["gaussian"][0]

        trellis_images = render_video(
            gauss,
            yaws=yaws,
            pitch=pitches,
            type="gaussian",
            bg_color=(1, 1, 1),
        )["color"]
        trellis_images_pil = [Image.fromarray(img) for img in trellis_images]
        return trellis_images_pil

    except Exception as e:
        print("Trellis error: ", e)
        return []


def init_results(dataset: str, mode: str) -> Dict:
    """
    Initialize the results dictionary for storing experiment outcomes.

    Args:
        dataset: the dataset being used ('objaverse' or 'co3d')
        mode: the mode of operation ('rank' or 'gt_prob')

    Returns:
        A dictionary to store results of the experiment.
    """
    results = {}
    if dataset == "objaverse" and mode == "gt_prob":  # Fig. 12
        results["correct_rot_by_rot_gt_prob"] = [[] for _ in range(10)]
        results["correct_trellis_baseline_by_rot_gt_prob"] = [[] for _ in range(10)]
        results["correct_focal_by_rot_gt_prob"] = [[] for _ in range(10)]
    else:
        # Fig 5 and 6 (Objaverse rank, and CO3D)
        results["correct_naive"] = []
        results["correct_focal"] = []

        if dataset == "co3d":  # Fig. 6 (CO3D)
            results["correct_trellis_baseline"] = []
            results["correct_random_rotation"] = []
    return results


def print_results(results: Dict, dataset: str, mode: str, num_samples: int) -> None:
    """
    Print the results of the experiment

    Args:
        results: dictionary containing the results of the experiment
        dataset: the dataset being used ('objaverse' or 'co3d')
        mode: the mode of operation ('rank' or 'gt_prob')
        num_samples: number of samples processed
    """
    print("Num Samples:", num_samples)
    if dataset == "objaverse" and mode == "gt_prob":  # Fig. 12
        processed_rot_by_rot_gt_prob = np.zeros((10))
        processed_tb_by_rot_gt_prob = np.zeros((10))
        processed_focal_by_rot_gt_prob = np.zeros((10))
        for i in range(10):
            if len(results["correct_rot_by_rot_gt_prob"][i]) > 0:
                processed_rot_by_rot_gt_prob[i] = np.mean(
                    np.array(results["correct_rot_by_rot_gt_prob"][i]), axis=0
                )
            if len(results["correct_trellis_baseline_by_rot_gt_prob"][i]) > 0:
                processed_tb_by_rot_gt_prob[i] = np.mean(
                    np.array(results["correct_trellis_baseline_by_rot_gt_prob"][i]),
                    axis=0,
                )
            if len(results["correct_focal_by_rot_gt_prob"][i]) > 0:
                processed_focal_by_rot_gt_prob[i] = np.mean(
                    np.array(results["correct_focal_by_rot_gt_prob"][i]), axis=0
                )
        print("Rotated by Rot GT Prob Accuracy:", processed_rot_by_rot_gt_prob)
        print("Trellis Baseline by Rot GT Prob Accuracy:", processed_tb_by_rot_gt_prob)
        print("Focal by Rot GT Prob Accuracy:", processed_focal_by_rot_gt_prob)
    else:
        # Fig. 5 and 6 (Objaverse rank, and CO3D)
        print("Naive Accuracy:", np.mean(np.array(results["correct_naive"]), axis=0))
        print("Focal Accuracy:", np.mean(np.array(results["correct_focal"]), axis=0))

        if dataset == "co3d":  # Fig. 6 (CO3D)
            print(
                "Trellis Baseline Accuracy:",
                np.mean(np.array(results["correct_trellis_baseline"]), axis=0),
            )
            print(
                "Random Rotation Accuracy:",
                np.mean(np.array(results["correct_random_rotation"]), axis=0),
            )


def run_3d(
    args: Args,
    data_cfg: DictConfig,
    gen_pose_tensor: torch.tensor,
    ovseg_classifier: OVSEGClassifier,
    clip_classifier: CLIPClassifier,
    diff_model: Optional[StableDiffusion] = None,
) -> None:
    """Function to run 3D viewpoint evaluation.

    Args:
        args: args object containing the script arguments
        data_cfg: data configuration object for Dataset
        gen_pose_tensor: [N x 2] tensor containing the N poses of [el, az] to generate
        ovseg_classifier: the OVSEG classifier
        clip_classifier: the CLIP classifier
        diff_model: the StableDiffusion model for energy computation
    """
    trellis_yaws = gen_pose_tensor[:, 1] * np.pi / 180
    trellis_pitch = gen_pose_tensor[:, 0] * np.pi / 180

    results = init_results(args.dataset, args.mode)

    assert args.dataset in ["objaverse", "co3d"], "Dataset must be objaverse or co3d"
    assert args.mode in ["rank", "gt_prob"], "Mode must be rank or gt_prob"
    assert args.mode == "gt_prob" or args.dataset == "objaverse", (
        "Rank mode is only supported for Objaverse dataset"
    )
    assert args.canon_2d_pattern == "0" or args.mode == "gt_prob", (
        "Canon 2D pattern of 5 is only supported in gt_prob mode"
    )

    if args.dataset == "objaverse":
        dataset = ObjaverseDataset(cfg=data_cfg)
    else:  # co3d
        dataset = CO3DDataset(cfg=data_cfg, thresh=args.co3d_thresh)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )
    classes = dataset.classes

    # load precomputed random indices and angles if in 'gt_prob' mode to help with reproduceability
    # rank_idxs helps select one random ranked view per object
    # rank_angles specifies a random angle that the image is rotated with at loading, before any processing.
    # NOTE: this rotation is NOT part of the algorithm. It is only for the gt_prob mode to evaluate what happens
    # if input Objaverse views had been tilted beforehand.
    if args.mode == "gt_prob":
        rank_idxs = np.load(args.random_ind_file)
        rank_angles = np.load(args.random_angle_file)

    # grab dataset specific rotation function
    rotate_fn = (
        rotate_image_objaverse if args.dataset == "objaverse" else rotate_image_co3d
    )

    # main loop over data
    for i, (img_cls, img_trl, label) in tqdm(
        enumerate(dataloader)
    ):  # img shape: [B, Num Views, C, H, W]
        num_views = img_cls.size(1)
        label = label[0].item()

        # Compute naive accuracy for figures 5 (Objaverse rank) and 6 (CO3D).
        if (
            args.dataset == "objaverse" and args.mode == "rank"
        ) or args.dataset == "co3d":
            this_naive_acc = []
            this_naive_gt_probs = []
            this_naive_energy = []
            this_naive_preds = []
            for j in range(num_views):
                img_torch = img_cls[0, j]  # shape: [C, H, W]
                img = torchvision.transforms.ToPILImage()(img_torch)

                # compute logits / energy
                ovseg_logits, naive_energy = classify_and_compute_energy(
                    args, [img], classes, ovseg_classifier, clip_classifier, diff_model
                )

                # bookkeeping
                this_naive_acc.append(ovseg_logits.argmax(dim=-1).item() == label)
                this_naive_gt_probs.append(
                    torch.nn.functional.softmax(ovseg_logits, dim=-1)[0, label].item()
                )
                this_naive_energy.append(naive_energy.item())
                this_naive_preds.append(ovseg_logits.argmax(dim=-1).item())

            # sort this object's naive results over ranks by gt probabilities.
            # Then, sort the accuracy and energy the same way to align them.
            _, sorted_indices = torch.sort(
                torch.tensor(this_naive_gt_probs), dim=0, descending=True
            )  # shape: [num views]
            sorted_acc = torch.gather(
                torch.tensor(this_naive_acc), dim=0, index=sorted_indices
            )  # shape: [num views]

            results["correct_naive"].append(sorted_acc.tolist())

        # collect the subset of views to evaluate the rest of the settings on
        if args.dataset == "objaverse":
            if args.mode == "rank":
                views_to_eval = [
                    int(sorted_indices[i]) for i in args.trellis.indices_to_gen
                ]
            else:  # gt_prob
                views_to_eval = [int(rank_idxs[i])]
        else:  # co3d
            views_to_eval = [0]

        # this image's results - save to aggregate over views
        this_correct_rotated = []
        this_energy_rotated = []
        this_pred_rotated = []
        this_correct_trellis_baseline = []
        this_correct_focal = []
        this_correct_random_rotation = []

        # iterate over the view subset to evaluate the rest of the modes. For the rank mode, this goes over the 4
        # plotted points. For the gt_prob mode, this is just one view per object. And for CO3D it's just the image.
        for j, idx in enumerate(views_to_eval):
            # maintain img and img_torch which are RGB with white background, and img_trellis and img_trellis_torch,
            # which are RGBA images for Objaverse, and preprocessed RGB images with black background for CO3D. Black
            # background because that's what TRELLIS' preprocessor does.
            img_torch = img_cls[0, idx]  # shape: [C, H, W]
            img = torchvision.transforms.ToPILImage()(img_torch)
            img_trellis_torch = img_trl[0, idx]  # shape: [C, H, W]
            img_trellis = torchvision.transforms.ToPILImage()(img_trellis_torch)

            # compute angle to rotate the input by before running any evaluation
            if (
                args.dataset == "objaverse"
                and args.random_angle_file != ""
                and args.mode == "gt_prob"
            ):
                rand_angle = int(rank_angles[i]) * -45
            else:
                rand_angle = 0

            # Compute accuracy on the pre-rotated inputs, if in gt mode for objaverse. Else, it is just 0 degree rotation
            rotated_img, rotated_img_trellis = rotate_fn(img, img_trellis, rand_angle)

            # Compute naive tilted accuracy
            rot_ovseg_logits, rot_energy = classify_and_compute_energy(
                args,
                [rotated_img],
                classes,
                ovseg_classifier,
                clip_classifier,
                diff_model,
            )
            this_correct_rotated.append(rot_ovseg_logits.argmax(dim=-1).item() == label)
            this_energy_rotated.append(rot_energy.item())
            this_pred_rotated.append(rot_ovseg_logits.argmax(dim=-1).item())
            rot_gt_prob = (
                torch.nn.functional.softmax(rot_ovseg_logits, dim=-1)[0, label].item()
                - 1e-6
            )  # avoid index overflowing later

            # Compute accuracy for the random rotation baseline for Fig. 6 (CO3D).
            # Randomly rotate the image, run through the classifier.
            if args.dataset == "co3d":
                random_rot_angle = random.uniform(-180, 180)
                random_rot_img, _ = rotate_fn(img, img_trellis, random_rot_angle)

                rand_rot_ovseg_logits, _ = classify_and_compute_energy(
                    args,
                    [random_rot_img],
                    classes,
                    ovseg_classifier,
                    clip_classifier,
                    diff_model,
                )
                this_correct_random_rotation.append(
                    rand_rot_ovseg_logits.argmax(dim=-1).item() == label
                )

            # Compute accuracy for the Trellis baseline for Figs. 6 (CO3D) and 12 (Objaverse gt_prob).
            # Generate TRELLIS image at precomputed best angle, run through classifier
            if (
                args.dataset == "objaverse" and args.mode == "gt_prob"
            ) or args.dataset == "co3d":
                # generate TRELLIS images at the grid of locations, classify, and compute energy
                trellis_baseline_images = generate_trellis_images(
                    args,
                    rotated_img_trellis,
                    torch.tensor([150 * np.pi / 180]),
                    torch.tensor([30 * np.pi / 180]),
                )
                if len(trellis_baseline_images) > 0:
                    tb_ovseg_logits, _ = classify_and_compute_energy(
                        args,
                        trellis_baseline_images,
                        classes,
                        ovseg_classifier,
                        clip_classifier,
                        diff_model,
                    )
                    this_correct_trellis_baseline.append(
                        tb_ovseg_logits.argmax(dim=-1).item() == label
                    )
                else:
                    # in case TRELLIS failed to run, due to OOM from generating way too many points in sampling, etc.,
                    # mark the image as incorrectly classified. < 1% of objects impacted.
                    this_correct_trellis_baseline.append(False)

            # Compute accuracy for FoCal: 1) apply 2D canon, 2) run TRELLIS over the selected pre-tilts, 3) pick the best at the end
            # 4) run 2D canon on the rotated image

            # In case the image is poorly tilted for TRELLIS, we want to use the 2D canonicalizer to see if we can optimally
            # rotate it in 2D before passing it to TRELLIS. We do this by generating a set of pre-tilted images, classifying them,
            # and picking the best energy one as the "center" of the range. The pattern specifies how many angles around the center
            # to try. If pattern is "0", then just use the center. If "5", then use center +/- 45 and +/- 90 degrees.
            assert args.canon_2d_pattern in ["0", "5"], (
                "Invalid canon 2d pattern, must be 0 or 5"
            )
            num_tilts = 1 if args.canon_2d_pattern == "0" else 8
            pre_tilted_imgs = []
            for this_tilt in range(num_tilts):
                tilt_angle = this_tilt * 45
                this_tilted_img, _ = rotate_fn(
                    rotated_img, rotated_img_trellis, tilt_angle
                )

                pre_tilted_imgs.append(this_tilted_img)

            pre_tilted_ovseg_logits, pre_tilted_energy = classify_and_compute_energy(
                args,
                pre_tilted_imgs,
                classes,
                ovseg_classifier,
                clip_classifier,
                diff_model,
            )
            _, _, pt_best_index = select_best(
                pre_tilted_ovseg_logits.argmax(dim=-1), pre_tilted_energy
            )

            # select the best random rotation pre-tilt
            best_angle = pt_best_index * 45
            pre_tilt_angles = get_pretilt_angles(args.canon_2d_pattern, best_angle)

            # Now that we have the best pretilt angle and the ones we want to sample around it,
            # run TRELLIS on all pretilts we have selected. Aggregate the best pred and energy from
            # each pretilt run into trellis_best_preds and trellis_best_energy, then select the best
            trellis_best_preds = []
            trellis_best_energy = []
            for pre_tilt_angle in pre_tilt_angles:
                # rotate the image according to the pretilt, generate TRELLIS images, classify, and compute energy
                this_img, this_img_trellis = rotate_fn(
                    rotated_img, rotated_img_trellis, pre_tilt_angle
                )

                # if in use_original mode, classify the pre-tilt input as a candidate
                if args.use_original:
                    this_ovseg_logits, this_energy = classify_and_compute_energy(
                        args,
                        [this_img],
                        classes,
                        ovseg_classifier,
                        clip_classifier,
                        diff_model,
                    )
                    trellis_best_preds.append(this_ovseg_logits.argmax(dim=-1).item())
                    trellis_best_energy.append(this_energy.item())

                # generate TRELLIS images at the grid of locations, classify, and compute energy
                trellis_images = generate_trellis_images(
                    args, this_img_trellis, trellis_yaws, trellis_pitch
                )
                if len(trellis_images) > 0:
                    trellis_ovseg_logits, trellis_energy = classify_and_compute_energy(
                        args,
                        trellis_images,
                        classes,
                        ovseg_classifier,
                        clip_classifier,
                        diff_model,
                    )
                    trellis_ovseg_preds = trellis_ovseg_logits.argmax(dim=-1)
                    trellis_best_pred, trellis_sorted_energy, _ = select_best(
                        trellis_ovseg_preds, trellis_energy
                    )
                    trellis_best_preds.append(trellis_best_pred)
                    trellis_best_energy.append(trellis_sorted_energy)

            # add the original input image as a candidate in case all the pre-tilts missed it.
            # NOTE: We grab from this_pred_rotated and this_energy_rotated, aka, the result from the image
            # after the input tilt from start of this j for loop, because that is simulating the input being
            # tilted from the real-world.
            if args.use_original:
                trellis_best_preds.append(this_pred_rotated[j])
                trellis_best_energy.append(this_energy_rotated[j])

            # Select the best over all of the pretilts as the FoCal prediction
            trellis_best_preds = torch.tensor(trellis_best_preds)
            trellis_best_energy = torch.tensor(trellis_best_energy)

            final_trellis_pred, _, _ = select_best(
                trellis_best_preds, trellis_best_energy
            )
            this_correct_focal.append(final_trellis_pred == label)

        # store results
        if args.dataset == "objaverse" and args.mode == "gt_prob":
            results["correct_rot_by_rot_gt_prob"][int(rot_gt_prob * 10 // 1)].append(
                this_correct_rotated
            )
            results["correct_trellis_baseline_by_rot_gt_prob"][
                int(rot_gt_prob * 10 // 1)
            ].append(this_correct_trellis_baseline)
            results["correct_focal_by_rot_gt_prob"][int(rot_gt_prob * 10 // 1)].append(
                this_correct_focal
            )
        else:
            # Fig 5 and 6 (Objaverse rank, and CO3D)
            results["correct_focal"].append(this_correct_focal)

            if args.dataset == "co3d":  # Fig. 6 (CO3D)
                results["correct_trellis_baseline"].append(
                    this_correct_trellis_baseline
                )
                results["correct_random_rotation"].append(this_correct_random_rotation)

        print_results(results, args.dataset, args.mode, i + 1)
    return


def main() -> None:
    """Main function to run the viewpoint experiments."""
    args = tyro.cli(Args)
    seed_everything(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_cfg = OmegaConf.load(args.dataset_cfg_file)

    ovseg_classifier = OVSEGClassifier(args.ovseg.config_file, args.ovseg.opts)
    # Load dataset and prompts
    prompts = load_prompts(args.dataset)
    prompts = [args.placeholdertoken.replace("[classname]", p) for p in prompts]
    if args.cls_energy.use_normalizing_prompt:
        prompts.append(args.cls_energy.normalizing_prompt)

    class_list = load_classes(args.dataset)
    clip_classifier = CLIPClassifier(
        prompts,
        prlc_ckpt=None,
        dataset=args.dataset,
        classes=class_list,
        ovseg_config="third_party_modified/ovseg/configs/ovseg_swinB_vitL_demo.yaml",
        ovseg_opts=[
            "MODEL.WEIGHTS",
            "third_party_modified/ovseg/ovseg_swinbase_vitL14_ft_mpt.pth",
        ],
    )

    use_diffusion = args.diffusion.factor > 0
    diffusion_model = None
    if use_diffusion:
        diffusion_model = StableDiffusion(
            model_id=args.diffusion.model,
            size=args.diffusion.size,
        )

    gen_pose_tensor = get_gen_grid(
        args.trellis.el_min,
        args.trellis.el_max,
        args.trellis.el_step,
        args.trellis.az_min,
        args.trellis.az_max,
        args.trellis.az_step,
    )

    run_3d(
        args,
        data_cfg,
        gen_pose_tensor,
        ovseg_classifier,
        clip_classifier,
        diffusion_model,
    )


if __name__ == "__main__":
    main()
