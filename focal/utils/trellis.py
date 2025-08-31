from typing import List, Tuple
import os

os.environ["SPCONV_ALGO"] = "auto"
from PIL import Image
from third_party_modified.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from third_party_modified.TRELLIS.trellis.utils.render_utils import (
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
    render_frames,
)
from third_party_modified.TRELLIS.trellis.representations.gaussian import Gaussian

pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()


def gen_3D_trellis(image: Image.Image, preprocess: bool = True) -> dict:
    """Helper function to run Trellis

    Args:
        image: PIL Image
        preprocess: whether to Preprocess image. Defaults to True.

    Returns:
        dict: outputs from Trellis
    """
    print("trellis bool", preprocess)
    outputs = pipeline.run(
        image, seed=1, preprocess_image=preprocess, formats=["gaussian"]
    )
    return outputs


def render_video(
    sample: Gaussian,
    yaws: List[float],
    pitch: List[float],
    resolution: int = 512,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    r: float = 2,
    fov: float = 40,
    **kwargs,
) -> List[Image.Image]:
    """Helper function to render video.

    Args:
        sample: Gaussian object
        yaws: List of yaw angles
        pitch: List of pitch angles
        resolution: resolution of the image. Defaults to 512.
        bg_color: background color. Defaults to (255, 255, 255).
        r: radius of the camera. Defaults to 2.
        fov: field of view. Defaults to 40.

    Returns:
        List[Image.Image]: List of rendered images
    """
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitch, r, fov
    )
    return render_frames(
        sample,
        extrinsics,
        intrinsics,
        {"resolution": resolution, "bg_color": bg_color},
        **kwargs,
    )
