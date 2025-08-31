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

## Basic organization:
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
In order to run 3D viewpoint experiments, you need to install detectron2 and TRELLIS. You also need to download and install patches for [Objaverse](https://github.com/allenai/objaverse-xl/tree/main), [TRELLIS](https://github.com/microsoft/TRELLIS.git), and [OVSeg](https://github.com/facebookresearch/ov-seg/tree/main) as part of the process in `scripts/download_3D_libraries.sh`:

```
conda create -n focal_3d python=3.10
conda activate focal_3d
pip install -r requirements.txt
./scripts/download_3D_libraries.sh
```

The OVSeg checkpoint also needs to be downloaded from their Google Drive: [https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view](https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view). Place it at: `./third_party/ovseg/ovseg_swinbase_vitL14_ft_mpt.pth`.

CO3D data would also have to be donwloaded from `https://ai.meta.com/datasets/co3d-downloads/` to run on CO3D. Structure should be: `datasets/co3d/orig/<class>/<uid>` when done.

For both Objaverse and CO3D, the datasets need to be filtered. Note that for CO3D, you need to export an OpenAI key, and for Objaverse, you have to install Blender and start the xserver as described in their repo ([https://github.com/allenai/objaverse-rendering](https://github.com/allenai/objaverse-rendering)):
```
python3 -m scripts.viewpoint_3D_process_objaverse
export OPENAI_API_KEY="<your key>"
python3 -m scripts.viewpoint_3D_process_co3d
```

To run the experiments after filtering:
```
python3 -m experiments.viewpoint_3D --mode rank --canon_2d_pattern 0 --dataset objaverse
python3 -m experiments.viewpoint_3D --mode gt_prob --canon_2d_pattern 5 --dataset objaverse
python3 -m experiments.viewpoint_3D --mode gt_prob --canon_2d_pattern 5 --dataset co3d 
```

Note that we ship modified copies of a few libraries in `third_party_modified` along with their appropriate licenses. We replaced the `README` in each sub-folder with a new one that discloses the changes made.

## How to cite

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
