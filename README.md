# install
<!-- conda create -y --prefix /home/jovyan/ovodov/.conda/stat_lr python=3.9 cupy pkg-config libjpeg-turbo opencv cudatoolkit=11.3 numba -c conda-forge -->
conda create -y --prefix /home/jovyan/ovodov/.conda/stat_lr python=3.9 cupy pkg-config libjpeg-turbo opencv -c conda-forge
conda activate /home/jovyan/ovodov/.conda/stat_lr
pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install numba ffcv pyyaml tensorboard