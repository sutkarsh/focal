"""Stable Diffusion model utilities for computing energy functions.

This module provides functionality to load and use Stable Diffusion models
for computing energy-based scores on images. Based on the Diffusion Classifier
codebase (https://github.com/diffusion-classifier/diffusion-classifier)
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# Model configuration
STABLE_DIFFUSION_MODEL_ID = "stabilityai/stable-diffusion-2-base"
ALLOWED_SIZES = [64, 128, 256, 512]

# Default parameters
DEFAULT_STEP_MIN = 500
DEFAULT_STEP_MAX = 1000
DEFAULT_STEP_STRIDE = 50
DEFAULT_BATCH_SIZE = 5
DEFAULT_NOISE_TENSORS = 50


class StableDiffusion:
    """Stable Diffusion model wrapper for energy-based scoring."""

    def __init__(
        self,
        model_id: str = STABLE_DIFFUSION_MODEL_ID,
        size: int = 256,
        custom_prompt: Optional[str] = None,
        num_noise_tensors: int = DEFAULT_NOISE_TENSORS,
        device: str = "cuda",
    ) -> None:
        """Initialize the Stable Diffusion model.

        Args:
            model_id: Model identifier for Stable Diffusion
            size: Size of input images (must be in ALLOWED_SIZES)
            custom_prompt: Optional custom prompt for text encoding
            num_noise_tensors: Number of noise tensors to pre-generate

        Raises:
            ValueError: If size is not in ALLOWED_SIZES
        """
        if size not in ALLOWED_SIZES:
            raise ValueError(f"Size must be one of {ALLOWED_SIZES}, got {size}")

        self.model_id = model_id
        self.size = size
        self.prompt = custom_prompt or ""
        self.device = device

        # Pre-generate noise tensors
        self.noise_tensor_list = [
            torch.randn(1000, 4, size // 8, size // 8).to(self.device).half()
            for _ in range(num_noise_tensors)
        ]

        # Initialize model components
        self._setup_pipeline()
        self._setup_text_encoder()
        self._setup_transform()

    def _setup_pipeline(self) -> None:
        """Setup the Stable Diffusion pipeline and scheduler."""
        # Initialize scheduler
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

        # Initialize pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, scheduler=scheduler, torch_dtype=torch.float16
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to(self.device)

        # Set models to eval mode and convert to half precision
        pipe.vae = pipe.vae.eval().half().to(self.device)
        pipe.unet = pipe.unet.eval().half().to(self.device)

        # Move scheduler parameters to GPU
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(self.device).half()

        self.pipe = pipe
        self.scheduler = scheduler

    def _setup_text_encoder(self) -> None:
        """Setup the text encoder and generate embeddings."""
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder.eval().to(self.device)

        # Tokenize prompt
        text_input = tokenizer(
            [self.prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Generate text embeddings
        with torch.inference_mode():
            self.text_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]

    def _setup_transform(self) -> None:
        """Setup image transformation pipeline."""
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def score(
        self,
        im: torch.Tensor,
        step_min: int = DEFAULT_STEP_MIN,
        step_max: int = DEFAULT_STEP_MAX,
        step_stride: int = DEFAULT_STEP_STRIDE,
        batchsize: int = DEFAULT_BATCH_SIZE,
        return_all_steps: bool = False,
        noise_tensor_idx: int = 0,
    ) -> np.ndarray:
        """Calculate diffusion loss for images at selected timesteps.

        Args:
            im: Input image tensor of shape (B, C, H, W)
            step_min: Minimum timestep to evaluate
            step_max: Maximum timestep to evaluate
            step_stride: Stride between timesteps
            batchsize: Batch size for processing
            return_all_steps: Whether to return losses for all steps
            noise_tensor_idx: Index of noise tensor to use

        Returns:
            Array of diffusion losses. If return_all_steps is True,
            shape is (B, num_steps), otherwise (B,)

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If CUDA runs out of memory
        """
        # Validate input parameters
        if step_min >= step_max:
            raise ValueError(
                f"step_min ({step_min}) must be less than step_max ({step_max})"
            )
        if step_stride <= 0:
            raise ValueError(f"step_stride must be positive, got {step_stride}")
        if batchsize <= 0:
            raise ValueError(f"batchsize must be positive, got {batchsize}")
        if noise_tensor_idx >= len(self.noise_tensor_list):
            raise ValueError(
                f"noise_tensor_idx ({noise_tensor_idx}) must be less than "
                f"number of noise tensors ({len(self.noise_tensor_list)})"
            )

        try:
            with torch.inference_mode():
                losses = []
                steps = (
                    torch.arange(step_min, step_max, step_stride).int().to(self.device)
                )
                B, C, H, W = im.shape
                n_steps = len(steps)

                # Convert image to half precision
                im = im.half()

                # Prepare indices and parameters
                im_idx = (
                    torch.arange(B)
                    .int()
                    .to(self.device)[:, None]
                    .repeat(1, n_steps)
                    .reshape(B * n_steps)
                )
                steps = steps[None, :].repeat(B, 1).reshape(-1)
                noise = self.noise_tensor_list[noise_tensor_idx][steps]
                text_input = self.text_embeddings.repeat(B * n_steps, 1, 1)
                alphas = (
                    self.scheduler.alphas_cumprod[steps]
                    .view(-1, 1, 1, 1)
                    .to(self.device)
                )

                # Process batches
                for i in range(0, B * n_steps, batchsize):
                    batch_slice = slice(i, i + batchsize)

                    # Prepare batch inputs
                    im_batch = im[im_idx[batch_slice]].to(self.device)
                    noise_batch = noise[batch_slice].to(self.device)
                    text_input_batch = text_input[batch_slice].to(self.device)
                    alphas_batch = alphas[batch_slice].to(self.device)
                    steps_batch = steps[batch_slice].to(self.device)

                    # Transform images
                    im_batch = self.transform(im_batch)

                    # Generate latents
                    latent_batch = self.pipe.vae.encode(im_batch).latent_dist.mean
                    latent_batch *= 0.18215

                    # Add noise to latents
                    noised_latent_batch = latent_batch * (
                        alphas_batch**0.5
                    ) + noise_batch * ((1 - alphas_batch) ** 0.5)

                    # Predict noise and calculate loss
                    noise_pred_batch = self.pipe.unet(
                        noised_latent_batch,
                        steps_batch,
                        encoder_hidden_states=text_input_batch,
                    ).sample

                    loss = (
                        F.mse_loss(noise_batch, noise_pred_batch, reduction="none")
                        .mean(dim=(1, 2, 3))
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    losses.append(loss)

                # Reshape and return results
                losses = np.concatenate(losses).reshape(B, n_steps)
                return losses if return_all_steps else losses.mean(axis=1)

        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    f"CUDA out of memory. Try reducing batch size (currently {batchsize}) "
                    "or using a smaller model/image size"
                ) from e
            raise


def test_diffusion_models() -> None:
    """Test the Stable Diffusion model with different configurations.

    This function tests the model with various image sizes and batch sizes
    to verify functionality and performance.
    """
    print("Testing Stable Diffusion model configurations...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create test image
    img = torch.randn(3, 3, 256, 256).to(device).half()

    for size in ALLOWED_SIZES:
        print(f"\nTesting size {size}:")
        try:
            diff = StableDiffusion(size=size, device=device)

            for batch_size in [1, 2, 5, 6]:
                try:
                    scores = diff.score(img, batchsize=batch_size)
                    print(f"  Batch size {batch_size}: scores shape {scores.shape}")
                except RuntimeError as e:
                    print(f"  Batch size {batch_size}: failed - {str(e)}")

        except Exception as e:
            print(f"  Failed to initialize model: {str(e)}")


if __name__ == "__main__":
    test_diffusion_models()
