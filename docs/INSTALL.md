## Installation

### Downloading data
- Downloading [data.zip](https://drive.google.com/file/d/1m5t7mAHt8-m5rLOxwd7M2DZKeqI2J6N6/view?usp=sharing) and unzip it. The data/ folder should looks like:
```
data/
├── hand_module
│   └── mean_mano_params.pkl
│   └── SMPLX_HAND_INFO.pkl
├── smplx
│   └── SMPLX_NEUTRAL.pkl
├── weights
│    └── hand_module
│    │       └── checkpoints_best/pose_shape_best.pth 
│    ├── hand_detector
     │       └── faster_rcnn_1_8_132028.pth  
     │       └── model_0529999.pth
│    ├── body_pose_estimator
│            └── checkpoint_iter_370000.pth
```
- Downloading [samples.zip](https://drive.google.com/file/d/1sX4HBeHQEE0LWbOXMjemHZizB_ZJM01p/view?usp=sharing) and unzip it. It contains sample images

### Requirements
- Linux with at least one GPU.
- Python ≥ 3.7
- CUDA >= 10.1
- PyTorch ≥ 1.4 and torchvision that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this  
- Other 3rd-party package: 
    - ```pip intall -r doc/requirements.txt``
- Detectron-2: [install](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
- Hand Detector: We use hand detector provided by [100DOH](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/download.html). Run following commands to install:
    - ```sh scripts/install_hand_detectors.sh ```
- 2D Body Pose estimator: Install with the following commands
    - ```sh scripts/install_pose2d.sh```


### Downloading data
- Our data (mano_mean_params.pkl)
- Our weights: weights/hand_module
- SMPL-X: downloading from their own website.
