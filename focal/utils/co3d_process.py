import glob
import json
import time
import os
from tqdm import tqdm
import base64
import logging
from io import BytesIO
from typing import Optional, Union, Dict, Tuple

import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.WARN, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

# Constants
FILTER_PROMPT = (
    "You are evaluating an image (left) and its segmentation overlay (right).\n"
    "Criteria:\n"
    "1) There is a single, clearly visible object that fits completely in the frame.\n"
    "2) The main object should not have any parts outside the frame. There should be a clear margin on all sides.\n"
    "3) The main object should be easily distinguishable and not blurry.\n"
    "4) The segmentation mask should accurately outline the main object.\n"
    "If all criteria are met, respond 'PASS' plus the main object. Otherwise, respond 'FAIL' plus a short reason.\n"
    "Your answer must start with 'PASS' or 'FAIL' and use fewer than 10 words."
)


def get_api_client(
    api_type: str = "openai",
) -> Optional[OpenAI]:
    """
    Get the appropriate API client with error handling.

    Args:
        api_type: Type of API client to return ('openai')

    Returns:
        API client object or None if initialization fails
    """
    if api_type == "openai":
        return OpenAI()
    else:
        raise ValueError(f"Unsupported API type: {api_type}")


def load_image(image_source: Union[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    Load an image from a file path or numpy array.

    Args:
        image_source: File path or numpy array containing the image

    Returns:
        Loaded image as numpy array, or None if loading fails
    """
    try:
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                logger.error(f"Image file not found: {image_source}")
                return None
            return np.array(Image.open(image_source).convert("RGB"))
        elif isinstance(image_source, np.ndarray):
            return image_source
        else:
            logger.error(f"Unsupported image source type: {type(image_source)}")
            return None
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None


def concat_and_base64_encode(
    image: np.ndarray, mask: np.ndarray, max_side: int = 512
) -> str:
    """
    Convert a numpy array image to base64 encoded string.

    Args:
        image: Input image as numpy array
        image: Mask as numpy array
        max_side: Maximum dimension for image resizing

    Returns:
        Base64 encoded string of the image

    Raises:
        ValueError: If image conversion fails
    """
    image = np.concatenate(
        [
            (image / image.max() * 255).astype(np.uint8),
            ((mask / mask.max() * 0.5 + image / image.max() * 0.5) * 255).astype(
                np.uint8
            ),
        ],
        axis=1,
    )
    try:
        img = Image.fromarray(image)

        # Scale down image if needed
        if max(img.size) > max_side:
            scale = max_side / max(img.size)
            new_size = tuple(int(dim * scale) for dim in img.size)
            img = img.resize(new_size)

        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img.save("test.png", format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_str
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")


def filter_segmentation(
    image: np.ndarray, mask: np.ndarray, filter_model: str = "openai"
) -> Tuple[bool, str]:
    """
    Filter image and segmentation mask pair using vision AI.

    Args:
        image: Input image as numpy array
        mask: Segmentation mask as numpy array
        filter_model: AI model to use for filtering

    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        # Get AI client
        client = get_api_client(filter_model)
        if client is None:
            return False, "Failed to initialize AI client"

        modelname = {
            "openai": "gpt-4o",
        }[filter_model]

        if filter_model == "openai":
            response = client.chat.completions.create(
                model=modelname,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": FILTER_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{concat_and_base64_encode(image, mask)}"
                                },
                            },
                        ],
                    }
                ],
            )
            result = response.choices[0].message.content.strip()
        else:
            return False, f"Unknown API type: {filter_model}"

        # Parse result
        is_valid = result.upper().startswith("PASS")
        reason = result.replace("PASS", "").replace("FAIL", "").strip()
        return is_valid, reason

    except Exception as e:
        logger.error(f"Segmentation filtering error: {str(e)}")
        return False, str(e)


def get_image_pairs(base_path: str) -> list[Tuple[str, str]]:
    """Get all numbered image pairs in the current directory.

    Args:
        base_path: Path to the directory containing images

    Returns:
        List of tuples containing image and corresponding segmentation mask paths
    """
    # Get all files matching pattern {number}.png
    base_images = glob.glob(os.path.join(base_path, "[0-9]*.png"))
    pairs = []

    for img in base_images:
        # Skip segmentation mask files (ones ending with 's.png')
        if img.endswith("s.png"):
            continue

        # Get the number from the filename
        num = os.path.basename(img).split(".")[0]
        seg_file = f"{num}s.png"

        # Get corresponding segmentation file path
        seg_file = os.path.join(base_path, f"{num}s.png")

        # Check if corresponding segmentation file exists
        if os.path.exists(seg_file):
            pairs.append((img, seg_file))

    return sorted(pairs)


def process_images(base_path: str, output_file: str = "results.json") -> None:
    """Process all image pairs and save results to JSON.

    Args:
        base_path: Path to the directory containing images
        output_file: Path to save the results JSON file
    """
    pairs = get_image_pairs(base_path)
    print(f"Found {len(pairs)} image pairs.")
    results: Dict[str, Dict] = {}

    # Load existing results if file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            results = json.load(f)

    for img_path, seg_path in tqdm(pairs):
        # Skip if already processed
        if img_path in results:
            print(f"Skipping already processed pair: {img_path}")
            continue

        print(f"Processing pair: {img_path}, {seg_path}")

        # Load images
        image = load_image(img_path)
        mask = load_image(seg_path)

        if image is None or mask is None:
            print(f"Error loading images for pair: {img_path}")
            continue

        # Run filtering
        is_valid, reason = filter_segmentation(image, mask)
        # print(f"{img_path}: {is_valid}. Reason: {reason}")

        # Store results
        results[img_path] = {
            "segmentation_file": seg_path,
            "status": "PASS" if is_valid else "FAIL",
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save results after each pair
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Sleep for 0.1 second -- to avoid spamming the API
        time.sleep(0.1)

    print(f"Processing complete. Results saved to {output_file}")
