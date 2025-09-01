# FoCal: Test-Time Canonicalization by Foundation Models for Robust Perception

[![arXiv](https://img.shields.io/badge/arXiv-2507.10375-b31b1b.svg)](https://arxiv.org/abs/2507.10375)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch2.4+-ee4c2c?logo=pytorch&logoColor=white)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)


This is the official repository for the paper "Test-Time Canonicalization by Foundation Models for Robust Perception" (ICML 2025).

### [Paper](https://arxiv.org/abs/2507.10375) | [Project Website](https://utkarsh.ai/projects/focal)

![FoCal](media/focal_teaser.png)

## Basic Organization:
- Each task is a python script of its own (in `experiments/` folder)
- Each python file imports from `focal/utils`, which contain most of the dependencies and library code.
- In particular, `focal/utils/energy.py` contains the energy function used for optimization.
- You can run the experiments using the following command:

```bash
python -m experiments.rotation_2D # Replace `rotation_2D` with the name of the task you want to run.
```

## Installation
The recommended way to install the code is using [uv](https://github.com/astral-sh/uv) package manager, however, we also provide a `requirements.txt` file for pip users.

We have tested the code with Python 3.9 and CUDA 12.8.

Recommended installation (uv):
```
pip install uv
uv sync
uv add . --dev
uv run -m experiments.rotation_2D # Run the 2D rotation experiment
```

pip installation:
```
pip install -r requirements.txt
pip install -e .
python -m experiments.rotation_2D
```

**Note**: You might need additional dependencies for specific tasks such as active vision (gaussian-splatting) and 3D viewpoints. See below for details.

### Gaussian Splatting for Active Vision
In order to run the active vision experiments, you need to download the Gaussian Splatting libraries and datasets. You can do this by running the following script:

```
# Install gaussian splatting dependencies
apt-get install python3-opencv # Install OpenCV if not already installed
source .venv/bin/activate # Activate the virtual environment (necessary for uv users)
bash scripts/download_gaussian_splatting_libraries.sh

# Download pretrained models and datasets
cd gaussian_splatting/
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip
unzip models.zip -d pretrained
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip -d data
```

### 3D Viewpoints
To run 3D viewpoint experiments, you need to install detectron2 and TRELLIS. You also need to download and install patches for [Objaverse](https://github.com/allenai/objaverse-xl/tree/main) (Apache 2.0), [TRELLIS](https://github.com/microsoft/TRELLIS.git) (MIT), and [OVSeg](https://github.com/facebookresearch/ov-seg/tree/main) (CC By-NC). Please refer to `Licenses and Third Parties` for more information.

#### Install
```
conda create -n focal_3d python=3.10
conda activate focal_3d
pip install -r requirements.txt
./scripts/download_3D_libraries.sh

### Apply OVSeg patch manually.
cd third_party_modified;
git clone https://github.com/facebookresearch/ov-seg.git
cd ov-seg;
git checkout 36f49d496714998058d115ffb6172d9d84c59065
git apply ../../patches/cc_by_nc/ovseg_patch.patch
cd ..;
mv ov-seg ovseg;
cd ..;
pip install -Ue third_party_modified/ovseg/third_party/CLIP/.
```

#### OVSeg (Classifier) Setup

The OVSeg checkpoint also needs to be downloaded from their Google Drive: [https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view](https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view). Place it at: `./third_party/ovseg/ovseg_swinbase_vitL14_ft_mpt.pth`.

#### Objaverse (Dataset) Setup
To run experiments on Objaverse, the dataset needs to be downloaded and filtered. To do so, install Blender and start the xserver (if on a headless server) as described in [https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering](https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering):

```
# Blender / xserver install
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz && \
  tar -xf blender-3.2.2-linux-x64.tar.xz && \
  rm blender-3.2.2-linux-x64.tar.xz
sudo apt-get install xserver-xorg -y && \
  sudo python3 start_x_server.py start

# Filter
python3 -m scripts.viewpoint_3D_process_objaverse
```

#### CO3D (Dataset) Setup
To run experiments on CO3D, the dataset needs to be downloaded and filtered. To download the data, download it from `https://ai.meta.com/datasets/co3d-downloads/`. Structure should be: `datasets/co3d/orig/<class>/<uid>` when done. To filter, export an OpenAI key and run the following:

```
export OPENAI_API_KEY="<your key>"
python3 -m scripts.viewpoint_3D_process_co3d
```

#### Run Experiments
```
python3 -m experiments.viewpoint_3D --mode rank --canon_2d_pattern 0 --dataset objaverse  # Fig. 5
python3 -m experiments.viewpoint_3D --mode gt_prob --canon_2d_pattern 5 --dataset objaverse  # Fig. 12
python3 -m experiments.viewpoint_3D --mode gt_prob --canon_2d_pattern 5 --dataset co3d  # Fig. 6
```

## How to Cite

If you find this code useful in your research, please cite this paper:

```bibtex
@inproceedings{
    singhal2025testtime,
    title={Test-Time Canonicalization by Foundation Models for Robust Perception},
    author={Utkarsh Singhal and Ryan Feng and Stella X. Yu and Atul Prakash},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=JMZ7mr19AK}
}
```

## Code Contributions

This repo has been created and is maintained by

- [Utkarsh Singhal](https://utkarsh.ai)
- [Ryan Feng](https://websites.umich.edu/~rtfeng/)

## Licenses and Third Parties
### MIT-Licensed Code
All original code in `confs/`, `experiments/`, `focal/`, `scripts/` is MIT-licensed. You may use, modify, and distribute it commercially.

### Patch Files

Some functionality requires patches for third-party libraries:

1. **TRELLIS (MIT)** – can be used commercially.  
2. **Objaverse (Apache 2.0)** (dataset) – can be used commercially with notices.  
3. **OVSeg (CC BY-NC 4.0)** (classifier for 3D viewpoint experiments) – **non-commercial only**.

Copies of their licenses / notices are included in `patches/`.

> The CC BY-NC patch is located at `patches/cc_by_nc/ovseg_patch.patch`.  
> Do **not** ship or include this patch in any commercial product.  
> You may apply it manually for non-commercial purposes only.  
