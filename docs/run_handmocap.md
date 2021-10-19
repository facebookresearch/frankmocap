# Hand Motion Capture Demo

Our hande module provides 3D hand motion capture output. We use the [HMR](https://akanazawa.github.io/hmr/) model, trained with several public hand pose datasets, the SOTA peformance among single-image based methods. See our [FrankMocap paper](https://penincillin.github.io/frank_mocap) for details.

<p>
    <img src="https://github.com/jhugestar/jhugestar.github.io/blob/master/img/frankmocap_hand.gif" height="256">
</p>


## A Quick Start
- Run the following. The mocap output will be shown on your screen
```
    # Using a machine with a monitor to show output on screen
    # OpenGL renderer is used by default (--renderer_type opengl)
    # The output images are also saved in ./mocap_output
    python -m demo.demo_handmocap --input_path ./sample_data/single_totalbody.mp4 --out_dir ./mocap_output

    # Screenless mode (e.g., a remote server)
    xvfb-run -a python -m demo.demo_handmocap --input_path ./sample_data/single_totalbody.mp4 --out_dir ./mocap_output

    # Set other render_type to use other renderers
    python -m demo.demo_handmocap --input_path ./sample_data/han_hand_short.single_totalbody.mp4 --out_dir ./mocap_output --renderer_type pytorch3d
```

## Run Demo with A Webcam Input
- Run,
    ```
        python -m demo.demo_handmocap --input_path webcam

        #or using opengl gui renderer
        python -m demo.demo_handmocap --input_path webcam --renderer_type opengl_gui
    ```
- See below to see how to control in opengl gui mode

## Run Demo for Egocentric Videos
- For 3D hand pose estimation in egocentric views, use --view_type ego_centric
    ```
    # with Screen
    python -m demo.demo_handmocap --input_path ./sample_data/han_hand_short.mp4 --out_dir ./mocap_output --view_type ego_centric

    # Screenless mode (e.g., a remote server)
    xvfb-run -a python -m demo.demo_handmocap --input_path ./sample_data/han_hand_short.mp4 --out_dir ./mocap_output --view_type ego_centric
    ```
- We use a different hand detector adjusted for egocentric views, but the 3D hand pose regressor is the same. By default, hand module assumes ```third_view```
<p>
    <img src="https://github.com/jhugestar/jhugestar.github.io/blob/master/img/frankmotion_egohand.gif" height="200">
</p>

## Other Renderer Options
- While opengl would be faster, it requires a screen connected to your machine. You may try screenless rendering or other rendering options described below.
- Screenless Opengl Rendering
    - If you do not have a screen attached in your machine (e.g., remote servers), use [xvfb-run](http://manpages.ubuntu.com/manpages/trusty/man1/xvfb-run.1.html) tool
    ```
        # The output images are also saved in ./mocap_output
        xvfb-run -a python -m demo.demo_handmocap --input_path ./sample_data/han_hand_short.mp4 --out_dir ./mocap_output --renderer_type opengl
    ```
- [Pytorch3D](https://pytorch3d.org/)
    - We use pytorch3d only for rendering purpose. 
    - Run the following command to use pytorch3d renderer
    ```
        python -m demo.demo_handmocap --input_path ./sample_data/han_hand_short.mp4 --out_dir ./mocap_output --renderer_type pytorch3d
    ```
- [OpenDR](https://github.com/mattloper/opendr/wiki)
    - Alternatively, run the following command to use opendr renderer
    ```
        python -m demo.demo_handmocap --input_path ./sample_data/han_hand_short.mp4 --out_dir ./mocap_output --renderer_type opendr
    ```

## Keys for OpenGL GUI Mode 
- In OpenGL GUI visualization mode, you can use mouse and keyboard to change view point. 
    - This mode requires a screen connected to your machine 
    - Keys in OpenGL 3D window
        - mouse-Left: view rotation
        - mouse-Right: view zoom chnages
        - shift+ mouseLeft: view pan
        - C: toggle for image view/3D free view
        - w: toggle wireframe/solid mesh
        - j: toggle skeleton visualization
        - R: automatically rotate views
        - f: toggle floordrawing
        - q: exit program


## Run Demo with Precomputed Bboxes 
- You can use the precomputed bboxes without running any detectors. Save bboxes for each image as a json format. Each json should contain the input image path.
- Assuming your bboxes are `/your/bbox_dir/XXX.json`
    ```
        python -m demo.demo_handmocap --input_path /your/bbox_dir --out_dir ./mocap_output
    ```
- Bbox format (json)
    ```
    {"image_path": "xxx.jpg", "hand_bbox_list":[{"left_hand":[x,y,w,h], "right_hand":[x,y,w,h]}], "body_bbox_list":[[x,y,w,h]]}
    ```
    - Note that bbox format is [minX,minY,width,height]
- For example
    ```
    {"image_path": "./sample_data/images/cj_dance_01_03_1_00075.png", "body_bbox_list": [[149, 380, 242, 565]], "hand_bbox_list": [{"left_hand": [288.9151611328125, 376.70184326171875, 39.796295166015625, 51.72357177734375], "right_hand": [234.97779846191406, 363.4115295410156, 50.28489685058594, 57.89691162109375]}]}
    ```
## Options 
### Input options
- `--input_path webcam`: Run demo for a video file  (without using `--vPath` option)
- `--input_path /your/path/video.mp4`: Run demo for a video file (mp4, avi, mov)
- `--input_path /your/dirPath`: Run demo for a folder that contains image seqeunces
- `--input_path /your/bboxDirPath`: Run demo for a folder that contains bbox json files. See [bbox format](https://github.com/facebookresearch/eft/blob/master/docs/README_dataformat.md#bbox-format-json)

- `--view_type`: The view type of input. It could be ```third_view``` or```ego_centric```


### Output options
- `--out_dir ./outputdirname`: Save the output images into files
- `--save_pred_pkl`: Save the pose reconstruction data (SMPL parameters and vertices) into pkl files   (requires `--out_dir ./outputdirname`)
- `--save_bbox_output`: Save the bbox data in json files (bbox_xywh format) (requires `--out_dir ./outputdirname`)
- `--no_display`: Do not visualize output on the screen

### Other options
- `--use_smplx`: Use SMPLX model for body pose estimation (instead of SMPL). This uses a different pre-trainined weights and may have different performance.
- `--start_frame 100 --end_frame 200`: Specify start and end frames (e.g., 100th frame and 200th frame in this example)
- `--single_person`: To enforce single person mocap (to avoid outlier bboxes). This mode chooses the biggest bbox. 

## License
- [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 

<!-- 

## Installation

### Basic Requirements
- Linux with at least one GPU.
- Python ≥ 3.7
- CUDA >= 10.0
- smplx >= 0.1.21
- PyTorch ≥ 1.4 and torchvision that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this  
- xvfb-run (for mesh rendering, it can be installed with apt-get)  
- Pytorch-3D: [Install](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
- Opendr ```pip install opendr```
- Other 3rd-party package: 
    - ```pip intall -r docs/requirements.txt``

### Download Extra Data
- Run the following script to download pretrained weight and others
    - ```sh scripts/download_data_hand_module.sh```
- The data will be downloaded in ```extra_data/hand_module```

### Setting Third-Party Required Data
- Download SMPLX Model (Neutral model: SMPLX_NEUTRAL.pkl):
    - Download in the original [SMPL-X website](https://smpl-x.is.tue.mpg.de/). You need to register to download the SMPLX data.
    - Put the ```SMPLX_NEUTRAL.pkl`` file in: ./extra_data/smpl/SMPLX_NEUTRAL.pkl

- Installing third-party hand bbox detection tools
    - Detectron-2: [install](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
    - Hand Detector: We use hand detector provided by [100DOH](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/download.html). Run following commands to install:
        - ```sh scripts/install_hand_detectors.sh ```
    - 2D Body Pose estimator: Install with the following commands
        - ```sh scripts/install_pose2d.sh```

### FYI, ./extra_data folder hierarchy
- The ./extra_data/ folder should look like:
```
extra_data/
├── hand_module
│   └── mean_mano_params.pkl
│   └── SMPLX_HAND_INFO.pkl
|   └── pretrained_weights
|   |   └── pose_shape_best.pth
│   └── hand_detector
│       └── faster_rcnn_1_8_132028.pth  
│       └── model_0529999.pth
├── body_module
|   └──body_pose_estimator
|       └── checkpoint_iter_370000.pth     
└── smpl
    └── SMPLX_NEUTRAL.pkl
``` -->
