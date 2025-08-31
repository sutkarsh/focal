"""Image classification models for 2D rotation tasks."""

from typing import List, Tuple

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from open_clip import create_model_and_transforms, get_tokenizer
from transformers import AutoModelForImageClassification
from focal.utils.equiadapt_classifier_utils import get_prediction_network


class CLIPClassifier:
    """CLIP-based image classifier."""

    def __init__(
        self, prompts: List[str], device: str = "cuda", *args, **kwargs
    ) -> None:
        """Initialize CLIP classifier.

        Args:
            prompts: List of text prompts for zero-shot classification
            device: Device to run model on
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)
        """
        self.device = device
        self.clip_model, self.clip_preprocess, self.clip_tokenizer = (
            self._setup_clip_model()
        )
        self.text_enc = self._encode_text(self.clip_model, self.clip_tokenizer, prompts)

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """Classify input image.

        Args:
            im: Input image tensor

        Returns:
            Classification logits
        """
        return self._classifier(
            im, self.clip_model, self.clip_preprocess, self.text_enc
        )

    def _setup_clip_model(self) -> Tuple[torch.nn.Module, transforms.Compose, object]:
        """Setup CLIP model components.

        Returns:
            Tuple of (model, preprocessing transforms, tokenizer)
        """
        clip_preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        clip_model, _, _ = create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        clip_model.eval().to(self.device)
        clip_tokenizer = get_tokenizer("ViT-H-14")
        return clip_model, clip_preprocess, clip_tokenizer

    def _encode_text(
        self, clip_model: torch.nn.Module, clip_tokenizer: object, prompts: List[str]
    ) -> torch.Tensor:
        """Encode text prompts using CLIP model.

        Args:
            clip_model: CLIP model
            clip_tokenizer: CLIP tokenizer
            prompts: List of text prompts

        Returns:
            Encoded text embeddings
        """
        with torch.inference_mode():
            batchsize = 100
            text_enc = torch.cat(
                [
                    clip_model.encode_text(
                        clip_tokenizer(prompts[i * batchsize : (i + 1) * batchsize]).to(
                            self.device
                        )
                    ).cpu()
                    for i in range(len(prompts) // batchsize + 1)
                ]
            )
            return F.normalize(text_enc, dim=-1)

    def _encode_im(
        self,
        im: torch.Tensor,
        clip_model: torch.nn.Module,
        clip_preprocess: transforms.Compose,
    ) -> torch.Tensor:
        """Encode image using CLIP model.

        Args:
            im: Input image tensor
            clip_model: CLIP model
            clip_preprocess: Preprocessing transforms

        Returns:
            Encoded image embeddings
        """
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        im_enc = torch.cat(
            [
                clip_model.encode_image(
                    clip_preprocess(im_).to(self.device).unsqueeze(0)
                ).cpu()
                for im_ in im
            ]
        )
        return im_enc

    def _classifier(
        self,
        im: torch.Tensor,
        clip_model: torch.nn.Module,
        clip_preprocess: transforms.Compose,
        text_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Classify image using CLIP model.

        Args:
            im: Input image tensor
            clip_model: CLIP model
            clip_preprocess: Preprocessing transforms
            text_enc: Encoded text embeddings

        Returns:
            Classification logits
        """
        with torch.inference_mode():
            im_enc = self._encode_im(im, clip_model, clip_preprocess)
            im_enc = F.normalize(im_enc, dim=-1)
            return im_enc @ text_enc.T


class DINOv2Classifier:
    def __init__(self, prompts, device: str = "cuda", *args, **kwargs) -> None:
        self.device = device
        self.model = AutoModelForImageClassification.from_pretrained(
            "facebook/dinov2-base-imagenet1k-1-layer"
        )
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, im, *args, **kwds):
        im = self.transform(im).to(self.device)
        return self.model(im).logits


class ResNet50:
    """ResNet-50 classifier pretrained on ImageNet."""

    def __init__(
        self, prompts: List[str], device: str = "cuda", *args, **kwargs
    ) -> None:
        """Initialize ResNet-50 classifier.

        Args:
            prompts: List of text prompts (must be length 1000 for ImageNet)
            device: Device to run model on
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Raises:
            AssertionError: If number of prompts is not 1000
        """
        assert len(prompts) == 1000, (
            "ResNet50 only supports 1000 classes (ImageNet). Your dataset is probably not ImageNet."
        )
        self.device = device
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """Classify input image.

        Args:
            im: Input image tensor

        Returns:
            Classification logits
        """
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        im = self.transform(im).to(self.device)
        with torch.inference_mode():
            return self.model(im).cpu()


class ViTB:
    """ViT-B/16 classifier pretrained on ImageNet."""

    def __init__(
        self, prompts: List[str], device: str = "cuda", *args, **kwargs
    ) -> None:
        """Initialize ViT-B/16 classifier.

        Args:
            prompts: List of text prompts (must be length 1000 for ImageNet)
            device: Device to run model on
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Raises:
            AssertionError: If number of prompts is not 1000
        """
        assert len(prompts) == 1000, (
            "ViT-B only supports 1000 classes (ImageNet). Your dataset is probably not ImageNet."
        )
        self.device = device
        self.model = torchvision.models.vit_b_16(pretrained=True)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """Classify input image.

        Args:
            im: Input image tensor

        Returns:
            Classification logits
        """
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        im = self.transform(im).to(self.device)
        with torch.inference_mode():
            return self.model(im).cpu()


class PRLC_R50:
    """PRLC's version of ResNet-50"""

    def __init__(
        self, prompts: List[str], device: str, ckpt: str, dataset: str, *args, **kwargs
    ) -> None:
        """Initialize ResNet-50 classifier.

        Args:
            prompts: List of text prompts
            device: Device to run model on
            ckpt: Path to pretrained checkpoint
            dataset: Dataset name (used for input shape and number of classes)
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Raises:
            AssertionError: If number of prompts is not 1000
        """
        self.device = device

        self.model = get_prediction_network(
            architecture="resnet50",
            dataset_name=dataset,
            use_pretrained=False,
            freeze_encoder=True,
            input_shape=(3, 32, 32),
            num_classes=len(prompts),
        ).to(device)

        prlc_dict = torch.load(ckpt, map_location=device)["state_dict"]
        prediction_network_params = {
            ".".join(k.split(".")[1:]): v
            for k, v in prlc_dict.items()
            if "prediction_network" in k
        }
        self.model.load_state_dict(prediction_network_params)

        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """Classify input image.

        Args:
            im: Input image tensor

        Returns:
            Classification logits
        """
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        im = self.transform(im).to(self.device)
        with torch.inference_mode():
            return self.model(im).cpu()


class PRLC_ViTB:
    """PRLC's version of ViT-B"""

    def __init__(
        self, prompts: List[str], device: str, ckpt: str, dataset: str, *args, **kwargs
    ) -> None:
        """Initialize ViT-B classifier.

        Args:
            prompts: List of text prompts
            device: Device to run model on
            ckpt: Path to pretrained checkpoint
            dataset: Dataset name (used for input shape and number of classes)
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Raises:
            AssertionError: If number of prompts is not 1000
        """
        self.device = device

        self.model = get_prediction_network(
            architecture="vit",
            dataset_name=dataset,
            use_pretrained=False,
            freeze_encoder=True,
            input_shape=(3, 32, 32),
            num_classes=len(prompts),
        ).to(device)

        prlc_dict = torch.load(ckpt, map_location=device)["state_dict"]
        prediction_network_params = {
            ".".join(k.split(".")[1:]): v
            for k, v in prlc_dict.items()
            if "prediction_network" in k
        }
        self.model.load_state_dict(prediction_network_params)

        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )

    def __call__(self, im: torch.Tensor) -> torch.Tensor:
        """Classify input image.

        Args:
            im: Input image tensor

        Returns:
            Classification logits
        """
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        im = self.transform(im).to(self.device)
        with torch.inference_mode():
            return self.model(im).cpu()


class _OVSEGArgs:
    """Helper class for OVSEG args"""

    def __init__(self, config_file: str, opts: List[str]) -> None:
        """Initialize _OVSEGArgs."""
        self.config_file = config_file
        self.opts = opts


class OVSEGClassifier:
    """OVSEG CLIP-based image classifier."""

    def setup_cfg(self, args: _OVSEGArgs):
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from third_party_modified.ovseg.open_vocab_seg import add_ovseg_config

        # from OVSEG's demo.py
        # load config from file and command-line arguments
        cfg = get_cfg()
        # for poly lr schedule
        add_deeplab_config(cfg)
        add_ovseg_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        return cfg

    def __init__(self, config_file: str, opts: List[str], *args, **kwargs) -> None:
        """Initialize OVSEG CLIP classifier.

        Args:
            config_file: Path to OVSEG config file
            opts: Additional options (e.g., model weights)
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)
        """
        from third_party_modified.ovseg.open_vocab_seg.utils import VisualizationDemo

        args = _OVSEGArgs(config_file, opts)
        cfg = self.setup_cfg(args)
        self._model = VisualizationDemo(cfg)

    def __call__(self, im: Image.Image, classes: List[str]) -> torch.Tensor:
        """Classify input image.

        Args:
            im: Input image (PIL), RGB
            classes: List of class names

        Returns:
            Classification logits
        """
        im = np.asarray(im)
        im = im[:, :, ::-1]  # ov-seg expects BGR
        return self._model.get_classification_clip(im, classes)[0]
