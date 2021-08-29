# Install ALL Modules without Pytorch3D
# CUDA 10.2, cuDNN 8
# PyTorch 1.9.0, TorchVision 0.10.0

FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${PATH}:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64


WORKDIR /root

# ninja-build is required by `detectron`
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates grep sed dpkg\
    git vim curl wget sudo\
    python3 python3-dev python3-setuptools python3-pip ninja-build\
    gcc-8 g++-8 cmake build-essential \
    ffmpeg libosmesa6-dev libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev freeglut3-dev xvfb && \
    rm /usr/bin/python; ln -s /usr/bin/python3 /usr/bin/python

RUN git clone https://github.com/facebookresearch/frankmocap.git /root/frankmocap

WORKDIR /root/frankmocap

RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.9.0 torchvision==0.10.0 scikit-build torchgeometry
RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r ./docs/requirements.txt
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html

RUN sh ./scripts/install_frankmocap.sh && \
    sh ./scripts/install_pose2d.sh && \
    sh scripts/download_data_body_module.sh && \
    sh scripts/download_data_hand_module.sh

CMD [ "/bin/bash" ]

# Docker image build finished
# Type the following command to enter the docker env:
# ```
# docker run -it --rm --gpus all frankmocap:latest
# ```

