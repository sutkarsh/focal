import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from typing import Tuple, List, Any
import torch

torch.use_deterministic_algorithms(True)
import random
from PIL import Image

import tyro
from dataclasses import dataclass, field
from pytorch_lightning import seed_everything
from third_party_modified.objaverse_xl.scripts.rendering.main import render_objects
import numpy as np
from tqdm import tqdm
from focal.utils.datasets import ObjaverseMetaDataset
from focal.utils.classifiers import OVSEGClassifier


@dataclass(frozen=True)
class ObjaverseArgs:
    """Configuration for 3d Viewpoint experiments.

    This class defines parameters for running viewpoint experiments where images
    from Objaverse-LVIS are rendered, and then aligned back via Trellis generations.
    """

    download_data: bool = True
    """whether to download the data or not"""

    quality_threshold: float = 0.1
    """Threshold for object quality. Minimum percent of renders that must be correct to pass."""

    dominance_threshold: int = 10
    """Threshold for object dominance. Minimum number of renders more that the GT class must be predicted
    compared to the next most common class"""

    overwrite: bool = False
    """Whether to overwrite existing Objaverse-LVIS renders or load if they exist"""

    path: str = "/home/rtfeng/.objaverse/hf-objaverse-v1"
    """path to object paths folder"""

    path_file: str = "/home/rtfeng/.objaverse/hf-objaverse-v1/object-paths.json.gz"
    """path to object paths file"""

    el_min: int = 10
    """Minimum elevation angle"""

    el_max: int = 90
    """Maximum elevation angle"""

    el_step: int = 30
    """Step size for elevation angle"""

    az_min: int = 10
    """Minimum azimuth angle"""

    az_max: int = 370
    """Maximum azimuth angle"""

    az_step: int = 30
    """Step size for azimuth angle"""

    folder_name: str = "datasets/objaverse/objaverse_lvis_renders_offset_10_original"
    """path to store the rendered images and outputs"""


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
class Args:
    """Configuration for 3d viewpoint Alignment Classifier experiments.

    This class defines parameters for running viewpoint experiments where images
    are rendered at various viewpoints and then aligned back using various methods.
    """

    seed: int = 42
    """Random seed for reproducibility"""

    objaverse: ObjaverseArgs = ObjaverseArgs()
    """Objaverse arguments"""

    ovseg: OVSEGArgs = OVSEGArgs()
    """OVSEG arguments"""


def is_img_bad(
    preds: torch.Tensor,
    this_object_label: int,
    num_renders: int,
    quality_threshold: float,
    dominance_threshold: float,
) -> bool:
    """Function to filter out bad Objaverse images based on the predictions over clean renders.

    Args:
        preds: the predictions in [num renders] shape
        this_object_label: the label of the object
        num_renders: the number of renders being evaluated
        quality_threshold: min percent of renders that must be correct
        dominance_threshold: min difference between the # of renders that are correct vs. any other class

    Returns:
        True if the object failed the filter test, False otherwise
    """
    correct = (preds == this_object_label).float()
    # quality filter. Flag images where the percentage of correct predictions is below the threshold
    if correct.sum() / num_renders < quality_threshold:
        return True

    # instance dominance filter. Flag images where there is a highly predicted, second confounding class
    try:
        highest_non_gt_label = preds[preds != this_object_label].mode().values.item()
        correct_non_gt = (preds == highest_non_gt_label).float()
        if correct.sum() - correct_non_gt.sum() < dominance_threshold:
            return True
    except IndexError:
        pass
    return False


class ObjaverseLVISDataset(torch.utils.data.Dataset):
    """Objaverse dataset."""

    def __init__(self, folder_name, obj_metadata) -> None:
        """Initializes the dataset.

        Args:
            folder_name: path to the folder containing the rendered images
            obj_metadata: object containing metadata from the Objaverse dataset
        """
        super().__init__()
        self.folder_name = folder_name
        self.obj_metadata = obj_metadata

    def __len__(self) -> int:
        return len(self.obj_metadata.uids_list)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """Get item from the dataset.

        Args:
            index: index of the item to get

        Returns:
            A tuple containing the path to the object's folder, the object's name, the object's label, and the object's UID"""
        this_object_path = os.path.join(
            self.folder_name,
            self.obj_metadata.corrected_uid_to_label_map[
                self.obj_metadata.uids_list[index]
            ],
            self.obj_metadata.uids_list[index],
        )
        this_object_name = self.obj_metadata.corrected_uid_to_label_map[
            self.obj_metadata.uids_list[index]
        ]
        this_object_uid = self.obj_metadata.uids_list[index]
        this_object_label = self.obj_metadata.corrected_classes.index(this_object_name)
        return this_object_path, this_object_name, this_object_label, this_object_uid


def render_and_filter_objaverse(
    obj_metadata: ObjaverseMetaDataset,
    ovseg_classifier: OVSEGClassifier,
    el_list: List[float],
    az_list: List[float],
    x_list: List[float],
    y_list: List[float],
    z_list: List[float],
    classes: List[str],
    args: Args,
) -> int:
    """Function to render and filter the Objaverse dataset.

    Args:
        obj_metadata: object containing metadata from the Objaverse dataset
        ovseg_classifier: the OVSEG classifier
        el_list: list of elevations to generate
        az_list: list of azimuths to generate
        x_list: list of x camera positions to generate
        y_list: list of y camera positions to generate
        z_list: list of z camera positions to generate
        classes: list of classes
        args: args object containing the script arguments

    Returns:
        The number of samples left after filtering.
    """

    def save_renders(this_object_path: str, target_dirs: List[str]) -> None:
        """Saves out the Objaverse renders from the default location to the processed data dir.

        Args:
            this_object_path: path to the folder where the renders will be saved
            target_dirs: list of directories where the renders were originally saved
        """
        for this_el, this_az, this_cam_x, this_cam_y, this_cam_z in zip(
            el_list, az_list, x_list, y_list, z_list
        ):
            path = os.path.join(
                target_dirs[0],
                f"{this_cam_x:.2f}_{this_cam_y:.2f}_{this_cam_z:.2f}.png",
            )
            image = Image.open(path)

            image.save(
                os.path.join(
                    this_object_path,
                    f"{this_el}_{this_az}_{this_cam_x:.2f}_{this_cam_y:.2f}_{this_cam_z:.2f}.png",
                )
            )

    def get_ovseg_scores(
        num_renders: int,
        this_object_path: str,
        this_object_name: str,
        this_object_uid: str,
    ):
        """Get OVSEG scores for the renders of a single object.

        Args:
            num_renders: number of renders to process
            this_object_path: path to the folder where the renders are saved
            this_object_name: name of the object
            this_object_uid: UID of the object

        Returns:
            this_ovseg_logits: OVSEG logits for the renders in [num_renders, num_classes] shape
        """
        this_ovseg_logits = torch.zeros((num_renders, len(classes)))
        # iterate over the rendered views
        for j, (this_el, this_az, this_cam_x, this_cam_y, this_cam_z) in enumerate(
            zip(el_list, az_list, x_list, y_list, z_list)
        ):
            path = os.path.join(
                this_object_path,
                f"{this_el}_{this_az}_{this_cam_x:.2f}_{this_cam_y:.2f}_{this_cam_z:.2f}.png",
            )

            out_prefix = f"{args.objaverse.folder_name}_outputs/{this_object_name}/{this_object_uid}/{this_el}_{this_az}"

            # if overwrite mode or the logits don't already exist, process. load the image, send to ovseg
            if args.objaverse.overwrite or not os.path.exists(
                f"{out_prefix}_logits.npy"
            ):
                os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
                img = load_obja_render(path)
                this_view_ovseg_logits = ovseg_classifier(img, classes)
                np.save(
                    f"{out_prefix}_logits.npy",
                    this_view_ovseg_logits.detach().cpu().numpy(),
                )
            else:
                this_view_ovseg_logits = torch.tensor(
                    np.load(f"{out_prefix}_logits.npy")
                )

            this_ovseg_logits[j, :] = this_view_ovseg_logits
        return this_ovseg_logits

    # bookkeeping
    os.makedirs(args.objaverse.folder_name, exist_ok=True)
    num_renders = len(el_list)
    num_samples = len(obj_metadata.uids_list)

    filtered_ovseg_logits = torch.zeros((num_samples, num_renders, len(classes)))
    filtered_cls_labels = torch.zeros((num_samples))
    list_file = open(f"{args.objaverse.folder_name}/filtered_objaverse_list.txt", "w")

    # go over each object. Render them, get the OVSeg predictions, filter. At the end, compute the rankings for each object.
    dataset = ObjaverseLVISDataset(args.objaverse.folder_name, obj_metadata)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
    )
    for i, (
        this_object_path,
        this_object_name,
        this_object_label,
        this_object_uid,
    ) in tqdm(enumerate(dataloader)):
        # generate and save the renders
        os.makedirs(this_object_path, exist_ok=True)

        if args.objaverse.overwrite or not os.path.exists(
            os.path.join(
                this_object_path,
                f"{el_list[-1]}_{az_list[-1]}_{x_list[-1]:.2f}_{y_list[-1]:.2f}_{z_list[-1]:.2f}.png",
            )
        ):
            print(i, this_object_name, this_object_uid)
            target_dirs = render_objects(
                [this_object_uid],
                obj_metadata.uid_to_label_map,
                obj_metadata.uid_to_object_paths,
                args.objaverse.path,
                "generate",
                x_list,
                y_list,
                z_list,
                False,
                num_renders=1,
            )

            save_renders(this_object_path, target_dirs)

        # get ovseg scores
        this_ovseg_logits = get_ovseg_scores(
            num_renders,
            this_object_path,
            this_object_name,
            this_object_uid,
        )

        # Filter
        preds = this_ovseg_logits.argmax(dim=-1)
        if is_img_bad(
            preds,
            this_object_label,
            num_renders,
            quality_threshold=args.objaverse.quality_threshold,
            dominance_threshold=args.objaverse.dominance_threshold,
        ):
            continue

        # bookkeeping / saving
        filtered_ovseg_logits[i] = this_ovseg_logits
        filtered_cls_labels[i] = this_object_label
        list_file.write(
            os.path.join(
                f"{args.objaverse.folder_name}",
                f"{obj_metadata.corrected_uid_to_label_map[obj_metadata.uids_list[i]]}",
                f"{obj_metadata.uids_list[i]}",
            )
            + "\n"
        )
        list_file.flush()
    list_file.close()

    return


def get_obja_locations(
    el_min: int = 0,
    el_max: int = 90,
    el_step: int = 30,
    az_min: int = 0,
    az_max: int = 360,
    az_step: int = 30,
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
    el_list = []
    az_list = []
    x_list = []
    y_list = []
    z_list = []
    for i in range(el_min, el_max, el_step):
        for j in range(az_min, az_max, az_step):
            el_list.append(i)
            az_list.append(j)
            el_rad = i * np.pi / 180
            az_rad = j * np.pi / 180
            x_list.append(2.2 * np.sin(az_rad) * np.cos(el_rad))
            y_list.append(-2.2 * np.cos(az_rad) * np.cos(el_rad))
            z_list.append(2.2 * np.sin(el_rad))

    return el_list, az_list, x_list, y_list, z_list


def load_obja_render(path: str, alpha: bool = False) -> Image.Image:
    """Load an Objaverse rendered image at the given path.

    Args:
        path: the location of the image
        alpha: whether to return a alpha channeled image or to remove it to RGB format

    Returns:
        The image in PIL format
    """
    image = Image.open(path)
    if alpha:
        return image
    image_np = np.array(image)
    alpha = image_np[:, :, 3:4] / 255.0
    white_im = np.ones_like(image_np) * 255.0
    image_np = alpha * image_np + (1.0 - alpha) * white_im
    image = Image.fromarray(image_np.astype(np.uint8)[:, :, :3])
    return image


def main() -> None:
    """Main function to run the viewpoint experiments."""
    args = tyro.cli(Args)
    seed_everything(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ovseg_classifier = OVSEGClassifier(args.ovseg.config_file, args.ovseg.opts)

    obj_metadata = ObjaverseMetaDataset(
        args.objaverse.path_file, args.objaverse.download_data
    )

    el_list, az_list, x_list, y_list, z_list = get_obja_locations(
        args.objaverse.el_min,
        args.objaverse.el_max,
        args.objaverse.el_step,
        args.objaverse.az_min,
        args.objaverse.az_max,
        args.objaverse.az_step,
    )

    render_and_filter_objaverse(
        obj_metadata=obj_metadata,
        ovseg_classifier=ovseg_classifier,
        el_list=el_list,
        az_list=az_list,
        x_list=x_list,
        y_list=y_list,
        z_list=z_list,
        classes=obj_metadata.corrected_classes,
        args=args,
    )


if __name__ == "__main__":
    main()
