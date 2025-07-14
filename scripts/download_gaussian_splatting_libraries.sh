git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive gaussian_splatting
cd gaussian_splatting/
git checkout ea68bdf29c3b11d1a7ec2e5a11b1af2c266bd7f2
pip install -e submodules/diff-gaussian-rasterization/
pip install -e submodules/simple-knn/
cd submodules/simple-knn/
git checkout f155ec04131cb579f53443a06879d37115f4612f
python setup.py build_ext --inplace
cd ../diff-gaussian-rasterization/
git checkout 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
python setup.py build_ext --inplace
# uv add opencv-python # Replace with pip install opencv-python if using pip
pip install opencv-python
cd ../../../