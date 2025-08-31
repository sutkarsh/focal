import os
import numpy as np
import torch
from PIL import Image
import subprocess
import csv
import cv2
import math
import shutil
from tqdm import tqdm
from torchvision import transforms
import matplotlib

matplotlib.use("Agg")
from pytorch_lightning import seed_everything
from focal.utils.classifiers import OVSEGClassifier
from focal.utils.co3d_process import process_images

MIN_HIGH_CONF_THRESH = 0.8
MAX_LOW_CONF_THRESHS = [0.5, 0.3]


def resize_and_pad(img, size=512):
    """Resize and pad an image to a square of given size.
    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W).
        size (int): Desired size for the square image.
    Returns:
        torch.Tensor: Resized and padded image tensor of shape (C, size, size).
    """
    # Pad image and masks to form a square
    _, h, w = img.shape
    max_dim = max(w, h)
    pad_w, pad_h = (max_dim - w) // 2, (max_dim - h) // 2
    padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)

    new_img = transforms.Pad(padding)(img)
    new_img = transforms.Resize(size)(new_img)
    return new_img


def square_and_save(root, data_root, co3d_proc_dir):
    """Helper function to resize and pad images and masks to square format.

    Args:
        root (str): Base directory where the data is stored.
        data_root (str): Path of the directory for the specific video to process.
        co3d_proc_dir (str): Directory to save the processed images and masks.
    Returns:

    """
    if not os.path.exists(
        os.path.join(root, co3d_proc_dir, "images")
    ) or not os.path.exists(os.path.join(root, co3d_proc_dir, "gt_masks")):
        # square images
        subprocess.run(["mkdir", "-p", os.path.join(root, co3d_proc_dir, "images")])
        subprocess.run(["mkdir", "-p", os.path.join(root, co3d_proc_dir, "gt_masks")])
        jpgs = [
            p
            for p in os.listdir(os.path.join(root, data_root, "images"))
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        jpgs.sort(key=lambda p: int(os.path.splitext(p)[0].split("frame")[-1]))
        the_size = None
        for jpg in jpgs:
            img = Image.open(os.path.join(root, data_root, "images", jpg))
            img_torch = transforms.ToTensor()(img)
            if the_size is None:
                the_size = max(img_torch.shape[1], img_torch.shape[2])
            new_img_torch = resize_and_pad(img_torch, the_size)
            new_img = transforms.ToPILImage()(new_img_torch)
            new_img.save(
                os.path.join(
                    root, co3d_proc_dir, "images", jpg.split("frame")[-1]
                ).replace(".jpg", ".png")
            )

        # square masks
        mask_jpgs = [
            p
            for p in os.listdir(os.path.join(root, data_root, "masks"))
            if os.path.splitext(p)[-1] in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        mask_jpgs.sort(key=lambda p: int(os.path.splitext(p)[0].split("frame")[-1]))
        for mask_jpg in mask_jpgs:
            mask = Image.open(os.path.join(root, data_root, "masks", mask_jpg))
            mask_torch = transforms.ToTensor()(mask)
            new_mask_torch = resize_and_pad(mask_torch, the_size)
            new_mask = transforms.ToPILImage()(new_mask_torch)
            new_mask.save(
                os.path.join(
                    root, co3d_proc_dir, "gt_masks", mask_jpg.split("frame")[-1]
                ).replace(".jpg", ".png")
            )

    # get filenames
    pngs = [
        p
        for p in os.listdir(os.path.join(root, co3d_proc_dir, "gt_masks"))
        if os.path.splitext(p)[-1] in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    pngs.sort(key=lambda p: int(os.path.splitext(p)[0]))
    filenames = [
        p
        for p in os.listdir(os.path.join(root, co3d_proc_dir, "images"))
        if os.path.splitext(p)[-1] in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    filenames.sort(key=lambda p: int(os.path.splitext(p)[0]))
    filenames = [os.path.join(co3d_proc_dir, "images", f) for f in filenames]

    return filenames


def copy_frames(txt_file: str, out_dir: str):
    """Copy frames from the given text file to the filtered directory.

    Args:
        txt_file (str): Path to the text file with image paths.
        out_dir (str): Directory to copy the filtered frames to.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(txt_file, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for i, row in enumerate(reader):
            print(row[0])
            image_crop = (
                row[0].replace("images", "images_cropped").replace(".jpg", ".png")
            )
            mask_crop = (
                row[0].replace("images", "masks_cropped").replace(".jpg", ".png")
            )

            shutil.copy(image_crop, f"{out_dir}/{i}.png")
            shutil.copy(mask_crop, f"{out_dir}/{i}s.png")


# read in CO3D classes
classes = []
with open("focal/utils/co3d_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

out_file_thresh1 = open(
    f"datasets/co3d/co3d_filtered_{MAX_LOW_CONF_THRESHS[0]}_input_paths.txt", "a"
)
out_file_thresh2 = open(
    f"datasets/co3d/co3d_filtered_{MAX_LOW_CONF_THRESHS[1]}_input_paths.txt", "a"
)


def run_video(data_root):
    root = os.getcwd()
    batch_size = 4
    seed_everything(42)
    co3d_proc_dir = data_root.replace("orig", "co3d_processed")

    filenames = square_and_save(root, data_root, co3d_proc_dir)

    num_imgs = len(filenames)
    num_batches = math.ceil(num_imgs / batch_size)
    label = filenames[0].split("/")[3]

    mask_sizes = []
    all_logits_plain = torch.zeros((num_imgs, len(classes)))

    # iterate over dataset, crop and mask according to TRELLIS preprocessing. Compute OVSeg logits for filtering
    for i in range(num_batches):
        print(num_batches, i, filenames[i * batch_size])
        imgs_pil = []
        for j in range(batch_size):
            if i * batch_size + j >= num_imgs:
                break
            try:
                img = Image.open(
                    filenames[i * batch_size + j]
                    .replace("images", "images_cropped")
                    .replace(".jpg", ".png")
                )
                mask = Image.open(
                    filenames[i * batch_size + j]
                    .replace("images", "masks_cropped")
                    .replace(".jpg", ".png")
                )
            except Exception:
                print("Image not found")
                img = Image.open(filenames[i * batch_size + j])
                img = img.convert("RGB")

                mask = Image.open(
                    filenames[i * batch_size + j]
                    .replace("images", "gt_masks")
                    .replace(".jpg", ".png")
                )  ### TODO
                mask = mask.convert("L")

                mask_np = np.array(mask)

                # estimate bounding box from mask, crop according to TRELLIS preprocessing
                x, y, w, h = cv2.boundingRect(mask_np.astype(np.uint8))
                center_x = x + w // 2
                center_y = y + h // 2
                size = max(w, h)
                size = int(size * 1.2)
                bbox = (
                    center_x - size // 2,
                    center_y - size // 2,
                    center_x + size // 2,
                    center_y + size // 2,
                )
                img = img.crop(bbox)
                img = img.resize((518, 518), Image.Resampling.LANCZOS)
                mask = mask.crop(bbox)
                mask = mask.resize((518, 518), Image.Resampling.LANCZOS)

                # save img crop and mask
                os.makedirs(
                    os.path.dirname(
                        filenames[i * batch_size + j].replace(
                            "images", "images_cropped"
                        )
                    ),
                    exist_ok=True,
                )
                img.save(
                    filenames[i * batch_size + j]
                    .replace("images", "images_cropped")
                    .replace(".jpg", ".png")
                )
                os.makedirs(
                    os.path.dirname(
                        filenames[i * batch_size + j].replace("images", "masks_cropped")
                    ),
                    exist_ok=True,
                )
                mask.save(
                    filenames[i * batch_size + j]
                    .replace("images", "masks_cropped")
                    .replace(".jpg", ".png")
                )

            # remove background, set to white for ovseg
            img_np = np.array(img)
            mask_np = np.array(mask)
            mask_sizes.append(mask_np.sum())
            img_np = np.where(mask_np[:, :, None] > 128, img_np, 255)
            img = Image.fromarray(img_np)
            imgs_pil.append(img)

        # compute and store OVSeg logits
        logits = torch.zeros((len(imgs_pil), len(classes)))
        for j, img in enumerate(imgs_pil):
            logits[j, :] = ovseg_classifier(img, classes)
            clip_file_name = (
                filenames[i * batch_size + j]
                .replace("images", "ovseg_outputs_gt_white")
                .replace(".png", ".npy")
            )
            if j == 0:
                os.makedirs(os.path.dirname(clip_file_name), exist_ok=True)
            np.save(clip_file_name, logits[j, :].cpu().numpy())

        all_logits_plain[i * batch_size : i * batch_size + len(imgs_pil)] = logits

    # get the ground truth probabilities for the label, filter
    clip_gt_probs = torch.nan_to_num(all_logits_plain.softmax(dim=-1))[
        :, classes.index(label)
    ]

    indices = np.arange(num_imgs)
    np.random.shuffle(indices)

    # If there is at least one frame with gt prob > 0.8, grab first random non-empty frame with gt prob < threshold
    if torch.max(clip_gt_probs) > MIN_HIGH_CONF_THRESH:
        if torch.min(clip_gt_probs) < MAX_LOW_CONF_THRESHS[1]:
            for i in indices:
                if clip_gt_probs[i] < MAX_LOW_CONF_THRESHS[1] and mask_sizes[i] > 0:
                    out_file_thresh2.write(f"{filenames[i]} {clip_gt_probs[i]}\n")
                    out_file_thresh2.flush()
                    break
        if torch.min(clip_gt_probs) < MAX_LOW_CONF_THRESHS[0]:
            for i in indices:
                if clip_gt_probs[i] < MAX_LOW_CONF_THRESHS[0] and mask_sizes[i] > 0:
                    out_file_thresh1.write(f"{filenames[i]} {clip_gt_probs[i]}\n")
                    out_file_thresh1.flush()
                    break

    return


# iterate over videos, process each one to grab one random frame below the threshold if the video is valid
vid_list = []
with open("focal/utils/co3d_all.csv", "r") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i < 1:
            continue
        vid_list.append((row[0], row[1]))

# filter dataset
ovseg_classifier = OVSEGClassifier(
    "third_party_modified/ovseg/configs/ovseg_swinB_vitL_demo.yaml",
    ["MODEL.WEIGHTS", "third_party_modified/ovseg/ovseg_swinbase_vitL14_ft_mpt.pth"],
)
for i, (this_cls, vid) in tqdm(enumerate(vid_list), total=len(vid_list)):
    run_video(f"datasets/co3d/orig/{this_cls}/{vid}")

# copy over just the filtered frames
for thresh in MAX_LOW_CONF_THRESHS:
    copy_frames(
        f"datasets/co3d/co3d_filtered_{thresh}_input_paths.txt",
        f"datasets/co3d/co3d_filtered_{thresh}",
    )


for thresh in MAX_LOW_CONF_THRESHS:
    process_images(
        f"datasets/co3d/co3d_filtered_{thresh}",
        f"datasets/co3d/jsons/co3d_filtered_{thresh}.json",
    )
