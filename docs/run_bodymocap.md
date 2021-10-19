# Body Motion Capture Demo

For our body mocap module, we use [HMR](https://akanazawa.github.io/hmr/) network architecture, by borrowing the implementation from [SPIN](https://github.com/nkolot/SPIN) with modifications. We trained the model with [EFT dataset](https://github.com/facebookresearch/eft), showing the SOTA peformance among single-image based methods.

<p>
    <img src="https://github.com/jhugestar/jhugestar.github.io/blob/master/img/eft_bodymocap.gif" height="256">
</p>

## A Quick Start
```
    # Using a machine with a monitor to show output on screen
    # OpenGL renderer is used by default (--renderer_type opengl)
    # The output images are also saved in ./mocap_output
    python -m demo.demo_bodymocap --input_path ./sample_data/han_short.mp4 --out_dir ./mocap_output

    # Screenless mode (e.g., a remote server)
    xvfb-run -a python -m demo.demo_bodymocap --input_path ./sample_data/han_short.mp4 --out_dir ./mocap_output

    # Set other render_type to use other renderers
    python -m demo.demo_bodymocap --input_path ./sample_data/han_short.mp4 --out_dir ./mocap_output --renderer_type pytorch3d

```

## Run Demo with a Webcam Input
- Run,
    ```
        python -m demo.demo_bodymocap --input_path webcam

        #or using opengl gui renderer
        python -m demo.demo_bodymocap --input_path webcam --renderer_type opengl_gui
    ```
- See below to see how to control in opengl gui mode


## Other Renderer Options
- While opengl would be faster, it requires a screen connected to your machine. You may try screenless rendering or other rendering options described below.
- Screenless Opengl Rendering
    - If you do not have a screen attached in your machine (e.g., remote servers), use [xvfb-run](http://manpages.ubuntu.com/manpages/trusty/man1/xvfb-run.1.html) tool
    ```
        # The output images are also saved in ./mocap_output
        xvfb-run -a python -m demo.demo_bodymocap --input_path ./sample_data/han_short.mp4 --out_dir ./mocap_output --renderer_type opengl
    ```
- [Pytorch3D](https://pytorch3d.org/)
    - We use pytorch3d only for rendering purpose. 
    - Run the following command to use pytorch3d renderer
    ```
        python -m demo.demo_bodymocap --input_path ./sample_data/han_short.mp4 --out_dir ./mocap_output --renderer_type pytorch3d
    ```
- [OpenDR](https://github.com/mattloper/opendr/wiki)
    - Alternatively, run the following command to use opendr renderer
    ```
        python -m demo.demo_bodymocap --input_path ./sample_data/han_short.mp4 --out_dir ./mocap_output --renderer_type opendr
    ```

## Keys for GUI Mode 
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
<p>
    <img src="https://github.com/jhugestar/jhugestar.github.io/blob/master/img/eft_gui_viewer_2.gif" height="256">
</p>

## Run Demo with Precomputed Bboxes 
- You can use the precomputed bboxes without running any detectors. Save bboxes for each image as a json format. Each json should contain the input image path.
- Assuming your bboxes are `/your/bbox_dir/XXX.json`
    ```
        python -m demo.demo_bodymocap --input_path /your/bbox_dir --out_dir ./mocap_output
    ```
- Bbox format (json)
    ```
    {"image_path": "xxx.jpg", "hand_bbox_list":[{"left_hand":[x,y,w,h], "right_hand":[x,y,w,h]}], "body_bbox_list":[[x,y,w,h]]}
    ```
    - Note that bbox format is [min_x, min_y, width, height]
- For example
    ```
    {"image_path": "./sample_data/images/cj_dance_01_03_1_00075.png", "body_bbox_list": [[149, 380, 242, 565]], "hand_bbox_list": [{"left_hand": [288.9151611328125, 376.70184326171875, 39.796295166015625, 51.72357177734375], "right_hand": [234.97779846191406, 363.4115295410156, 50.28489685058594, 57.89691162109375]}]}
    ```

## Options 
### Input Options
- `--input_path webcam`: Run demo for a video file  (without using `--vPath` option)
- `--input_path /your/path/video.mp4`: Run demo for a video file (mp4, avi, mov)
- `--input_path /your/dirPath`: Run demo for a folder that contains image seqeunces
- `--input_path /your/bboxDirPath`: Run demo for a folder that contains bbox json files. See [bbox format](https://github.com/facebookresearch/eft/blob/master/docs/README_dataformat.md#bbox-format-json)

### Output Options
- `--out_dir ./outputdirname`: Save the output images into files
- `--save_pred_pkl`: Save the pose reconstruction data (SMPL parameters and vertices) into pkl files   (requires `--out_dir ./outputdirname`)
- `--save_bbox_output`: Save the bbox data in json files (bbox_xywh format) (requires `--out_dir ./outputdirname`)
- `--no_display`: Do not visualize output on the screen
- `--save_mesh`: Saving vertices and faces when save predicting results (otherwise, only smpl/smplx parameters are saved)

### Other Options
- `--use_smplx`: Use SMPLX model for body pose estimation (instead of SMPL). This uses a different pre-trainined weights and may have different performance.
- `--start_frame 100 --end_frame 200`: Specify start and end frames (e.g., 100th frame and 200th frame in this example)
- `--single_person`: To enforce single person mocap (to avoid outlier bboxes). This mode chooses the biggest bbox. 
<!-- - (TODO) `--download --url https://videourl/XXXX`: download public videos via `youtube-dl` and run with the downloaded video. (need to install youtube-dl first) -->

## Mocap Output Format (pkl)
As output, the 3D pose estimation data per frame is saved as a pkl file (with ```--pklout`` option). Each person's pose data is saved as follows:
```
'demo_type': ['body', 'hand', 'frank']
'smpl_type': ['smplx', 'smpl']
'pred_body_pose':  body pose parameters in angle-axis format # (24, 3, 3)
'pred_left_hand_pose': hand pose parameters in angle-axis format # (16, 3, 3)
'pred_betas': shape paramters # (10,)
'pred_camera':  #[cam_scale, cam_offset_x,, cam_offset_y ]
'pred_hand_bbox': bounding box for hand # {left_hand:[x,y,w,h], right_hand:[x,y,w,h]} or None
'pred_body_bbox': bounding box for body # [x,y,w,h]
'pred_vertices_smpl': # Original vertices from SMPL output
'pred_vertices_img': #3D SMPL vertices where X,Y are aligned to input image
'pred_joints_img': #3D joints where X,Y are aligned to input image
```

## Load Saved Mocap Data (pkl file)
- Run the following code to load and visualize saved mocap data files
```
#./mocap_output/mocap is the directory where pkl files exist
python -m  demo.demo_loadmocap --pkl_dir ./mocap_output/mocap
```
- Note: current version uses GUI mode for the visualization (requiring a screen). 
- The current mocap output is redundant, and there are several options to visualize meshes from them

```
if False:    #One way to visualize SMPL from saved vertices
    tempMesh = {'ver': pred_vertices_imgspace, 'f':  smpl.faces}
    meshList=[]
    skelList=[]
    meshList.append(tempMesh)
    skelList.append(pred_joints_imgspace.ravel()[:,np.newaxis])  #(49x3, 1)

    visualizer.visualize_gui_naive(meshList, skelList)

elif False: #Alternative way from SMPL parameters
    pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,[0] ], pose2rot=False)
    pred_vertices = pred_output.vertices
    pred_joints_3d = pred_output.joints
    pred_vertices = pred_vertices[0].cpu().numpy()
    
    tempMesh = {'ver': pred_vertices_imgspace, 'f':  smpl.faces}
    meshList=[]
    skelList=[]
    body_bbox_list=[]
    meshList.append(tempMesh)
    skelList.append(pred_joints_imgspace.ravel()[:,np.newaxis])  #(49x3, 1)
    visualizer.visualize_gui_naive(meshList, skelList)

else: #Another alternative way using a funtion
    
    smpl_pose_list =  [ pred_rotmat[0].numpy() ]        #build a numpy array
    visualizer.visualize_gui_smplpose_basic(smpl, smpl_pose_list ,isRotMat=True )       #Assuming zero beta
```

## License
- [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 



<!-- 
### Setting SMPL Model
- Download SMPL Model (Neutral model: basicModel_neutral_lbs_10_207_0_v1.0.0.pkl):
    - Download in the original [website](http://smplify.is.tue.mpg.de/login). You need to register to download the SMPL data.
    - Put the file in: ./extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

- (Optional) Download SMPLX Model (Neutral model: SMPLX_NEUTRAL.pkl):
    - You can use SMPL-X model for body mocap instead of SMPL model. 
    - Download ```SMPLX_NEUTRAL.pkl``` in the original [SMPL website](https://smpl-x.is.tue.mpg.de/). You need to register to download the SMPLX data.
    - Put the ```SMPLX_NEUTRAL.pkl`` file in: ./extra_data/smpl/SMPLX_NEUTRAL.pkl

### Installing third-party tools for body bbox detection
- Our 3D body mocap demo assumes that a bounding box is given for each person. For this, you need either of the following options.
- (Option 1) Use [2D keypoint detector](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch): 
    - Run the following script
    ```
    sh scripts/install_pose2d.sh
    ```
    - This will download the detector in ./detectors/body_pose_estimator
- (Option 2) Use your own bbox detection to pre-compute bboxes, and load from the exported bbox file. See the [bbox format](https://github.com/facebookresearch/eft/blob/master/docs/README_dataformat.md#bbox-format-json). -->
