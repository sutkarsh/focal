pip install 'git+https://github.com/facebookresearch/detectron2.git@9604f5995cc628619f0e4fd913453b4d7d61db3f'

mkdir third_party_modified;
cd third_party_modified;
git clone https://github.com/microsoft/TRELLIS.git
cd TRELLIS;
git checkout ab1b84a18ecc6610b2656026f78866aa2643631b
git apply ../../patches/trellis_patch.patch
cd ..
git clone https://github.com/facebookresearch/ov-seg.git
cd ov-seg;
git checkout 36f49d496714998058d115ffb6172d9d84c59065
git apply ../../patches/ovseg_patch.patch
cd ..
git clone https://github.com/allenai/objaverse-xl.git
cd objaverse-xl;
git checkout 7e05a89db4d9eabfdf4c8e6d9a99973716f5f61a
git apply ../../patches/objaverse_patch.patch
cd ..;
mv ov-seg ovseg;
cd ..;

pip install -Ue third_party_modified/ovseg/third_party/CLIP/.
cd third_party_modified/TRELLIS
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu121.html
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast # from TRELLIS install instructions
