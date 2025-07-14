"""
NeRF Dataset implementation using Gaussian Splatting for 3D scene representation.
This module provides functionality to load and render different viewpoints of 3D scenes.
"""

import os
from typing import List, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from argparse import ArgumentParser, Namespace

# Ugly PYPTHONPATH hack to avoid patching gaussian splatting's imports
import sys

HERE = os.path.dirname(__file__)  # /fmc_clean/experiments
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # /fmc_clean

GAUSS = os.path.join(ROOT, "gaussian_splatting")
sys.path.insert(0, GAUSS)

KNN_SRC = os.path.join(GAUSS, "submodules", "simple-knn")
sys.path.insert(0, KNN_SRC)

DIFFRAST_SRC = os.path.join(GAUSS, "submodules", "diff-gaussian-rasterization")
sys.path.insert(0, DIFFRAST_SRC)
###

from gaussian_splatting.scene import Scene
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render, GaussianModel
from gaussian_splatting.arguments import ModelParams, PipelineParams


class NeRFDataset(Dataset):
    """
    A PyTorch Dataset for handling Neural Radiance Fields (NeRF) data using Gaussian Splatting.

    This dataset loads 3D scenes and generates different viewpoints by varying camera positions.
    It supports rendering from different azimuth and elevation angles for each scene.
    """

    def __init__(
        self,
        root: str,
        labels: str = "objectron",
        device: str = "cuda",
        transform: Optional[Any] = None,
        views_per_scene: int = 100,
        num_viewpoints: int = 1,
        azimuth_range: float = 45,
        elevation_range: float = 10,
        num_labels: int = 9,
    ):
        """
        Initialize the NeRF dataset.

        Args:
            root: Root directory containing scene data
            labels: Type of labels to use (default: 'objectron')
            device: Device to use for computations (default: 'cuda')
            transform: Optional transform to be applied to rendered images
            views_per_scene: Number of different views to generate per scene
            num_viewpoints: Number of viewpoints to return per item (currently only 1 supported)
            azimuth_range: Range of azimuth angles for camera positions (in degrees)
            elevation_range: Range of elevation angles for camera positions (in degrees)
            num_labels: Number of different scene labels
        """
        self.root = root
        self.data = []
        self.targets = []
        self.num_viewpoints = num_viewpoints
        assert self.num_viewpoints == 1  # Currently only single viewpoint is supported
        self.num_labels = num_labels

        # Camera view parameters
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.views_per_scene = views_per_scene

        # Available scenes for rendering
        self.scenes = [
            "bonsai",
            "room",
            # "counter",
            "garden",
            "stump",
            "bicycle",
            # "flowers",
            "kitchen",
            # "treehill", # The commented out scenes seem to cause problems in some diff-gaussian-rasterization installs. TODO fix
        ]

        self.transform = transform
        self.device = device

        self.currently_loaded_scene = None
        self._refresh_parser()

    def _refresh_parser(self) -> None:
        """Initialize argument parser for Gaussian Splatting configuration."""
        self.parser = ArgumentParser(description="Testing script parameters")
        self.parser.add_argument("--scene", default="garden", type=str)
        self.parser.add_argument("--iteration", default=-1, type=int)
        self.parser.add_argument("--skip_train", default=True, type=bool)
        self.parser.add_argument("--quiet", action="store_true")

    def _get_scene_args(self, scene_name: str, root: str = "./") -> Namespace:
        """
        Get command line arguments for a specific scene.

        Args:
            scene_name: Name of the scene to load
            root: Root directory containing scene data

        Returns:
            Namespace containing merged arguments from command line and config file
        """
        assert scene_name in self.scenes, (
            f"Scene {scene_name} not found in available scenes"
        )

        # Set up command line arguments
        args_cmdline = self.parser.parse_args(
            args=[
                "--model_path",
                "./pretrained/stump",
                "-s",
                "data/stump/",
                "--data_device",
                "cuda",
            ]
        )

        # Update scene-specific paths
        args_cmdline.scene = scene_name
        args_cmdline.model_path = f"{root}/pretrained/{scene_name}"
        args_cmdline.source_path = f"{root}/data/{scene_name}"
        args_cmdline.skip_train = True
        args_cmdline.data_device = "cuda"

        # Load and merge with config file arguments
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        with open(cfgfilepath) as cfg_file:
            cfgfile_string = cfg_file.read()

        args_cfgfile = eval(cfgfile_string)
        merged_dict = vars(args_cfgfile).copy()
        for k, v in vars(args_cmdline).items():
            if v is not None:
                merged_dict[k] = v

        return Namespace(**merged_dict)

    def _load_scene(self, scene_name: str, root: str = "./") -> None:
        """
        Load a scene and prepare it for rendering.

        Args:
            scene_name: Name of the scene to load
            root: Root directory containing scene data
        """
        # Skip if scene is already loaded
        if self.currently_loaded_scene == scene_name:
            return

        self.currently_loaded_scene = scene_name
        self._refresh_parser()

        # Initialize model and pipeline parameters
        model = ModelParams(self.parser, sentinel=True)
        pipeline = PipelineParams(self.parser)
        args = self._get_scene_args(scene_name, root=root)

        # Set up scene and gaussians
        dataset = model.extract(args)
        iteration = args.iteration
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Set background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Calculate camera parameters
        view = scene.getTrainCameras()[0]
        average_up_dir = -torch.mean(
            torch.stack(
                [v.world_view_transform.T[1, :3] for v in scene.getTrainCameras()]
            ),
            dim=0,
        )

        ray_normalized = view.world_view_transform.T[2, :3] / torch.norm(
            view.world_view_transform.T[2, :3]
        )
        project_origin_lambda = torch.dot(-view.camera_center, ray_normalized)
        new_origin = view.camera_center + project_origin_lambda * ray_normalized
        new_radius = project_origin_lambda * 1.0
        current_cam_center = view.camera_center.unsqueeze(0)
        new_up = average_up_dir.unsqueeze(0)

        # Compute base azimuth and elevation
        base_az, base_el = self.compute_az_el(new_origin, current_cam_center.squeeze(0))

        # Store scene parameters
        self.base_az = base_az
        self.base_el = base_el
        self.new_radius = new_radius
        self.new_origin = new_origin
        self.new_up = new_up
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.view = view

    @staticmethod
    def compute_az_el(
        origin: torch.Tensor, point: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute azimuth and elevation angles between origin and point.

        Args:
            origin: Origin point in 3D space
            point: Target point in 3D space

        Returns:
            Tuple of (azimuth, elevation) angles in degrees
        """
        ray = point - origin
        dist = torch.norm(ray)
        el = -torch.asin(ray[1] / dist) * 180 / np.pi
        az = torch.atan2(ray[0], -ray[2]) * 180 / np.pi
        return az, el

    def render_view(self, az_added: float, el_added: float) -> torch.Tensor:
        """
        Render a view of the scene from specified azimuth and elevation offsets.

        Args:
            az_added: Additional azimuth angle in degrees
            el_added: Additional elevation angle in degrees

        Returns:
            Rendered image as a tensor
        """
        assert self.currently_loaded_scene is not None, (
            "No scene loaded. Call _load_scene first"
        )

        newR, newT = np.eye(3), np.zeros(3)

        newView = Camera(
            colmap_id=self.view.uid,
            R=newR,
            T=newT,
            FoVx=self.view.FoVx,
            FoVy=self.view.FoVy,
            image=self.view.original_image,
            gt_alpha_mask=None,
            image_name=self.view.image_name,
            uid=self.view.uid,
            data_device=self.view.data_device,
        )

        with torch.no_grad():
            rendering = render(newView, self.gaussians, self.pipeline, self.background)[
                "render"
            ]
        return rendering

    def __len__(self) -> int:
        """Return total number of items in dataset."""
        return len(self.scenes) * self.views_per_scene

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Get a batch of rendered views for a scene.

        Args:
            idx: Index of the item to get

        Returns:
            Tuple containing:
            - Rendered images tensor [N x C x H x W]
            - Frame IDs tensor [N]
            - Scene IDs tensor [N]
            - List of scene names [N]
            where N is num_viewpoints
        """
        scene_id = idx // self.views_per_scene
        scene_name = self.scenes[scene_id]
        self._load_scene(scene_name, root=self.root)

        # Generate random camera angles within specified ranges
        az = torch.rand(1).item() * self.azimuth_range * 2 - self.azimuth_range
        el = torch.rand(1).item() * self.elevation_range * 2 - self.elevation_range

        img = self.render_view(az, el)

        if self.transform is not None:
            img = self.transform(img)

        assert self.num_viewpoints == 1

        # Repeat single viewpoint data for consistency
        imgs = img.unsqueeze(0).repeat(self.num_viewpoints, 1, 1, 1)
        scene_names = [scene_name for _ in range(self.num_viewpoints)]
        frame_id = idx % self.views_per_scene
        frame_ids = torch.tensor([frame_id for _ in range(self.num_viewpoints)])
        scene_ids = torch.tensor([scene_id for _ in range(self.num_viewpoints)])

        return imgs, frame_ids, scene_ids, scene_names
