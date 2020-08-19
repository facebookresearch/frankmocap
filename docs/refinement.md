## Temporal refinement

In this script, I will use data put in /checkpoint/rongyu/data/3d_hand/demo_data/youtube_example as example.  Let's set $root_dir = /checkpoint/rongyu/data/3d_hand/demo_data/youtube_example

To begin with, please check $root_dir/frame/ and $root_dir/openpose_output/ directory. The extracted frames and openpose results are put in these two directories. The data is formed by the sequence (one single demo video).

```
frame/
    demo_sequence_01/
    demo_sequence_02/
    demo_sequence_03/

openpose_output/
    demo_sequence_01/
    demo_sequence_02/
    demo_sequence_03/
```

### Prepare hand image

To start with, visualize the openpose output and crop hands  
```
sh script/prepare_hand_image.sh
```
The results of visualization are stored in $root_dir/openpose_visualization.  
The results of cropped hands are stored in $root_dir/image_hand.  
The results of bbox use to cropped hands are stored in $root_dir/bbox_info.  


### Augment Hand BBOX
Openpose has some missing detection for some sampels, this script is used to copy and paste missing bbox from previous frame.
```
sh script/augment_bbox.sh
```
The $root_dir/bbox_info and $root_dir/image_hand was updated with new bbox and hand images.



### Get initial prediction

Please change the $demo_img_dir in script/demo.sh to demo_data/youtube_example/image_hand and run
```
sh script/demo.sh
```
Also change the %test_checkpoint_dir if you want to use other weights to test. By default, it will use the best weights storted in model/checkpoints_best.  

After running complete, you can get predicted results stored in model/evaluate_results/pred_results_demo_youtube_example_pose_shape_best.pkl  
Please copy this file to $root_dir/prediction/pred_results.pkl (mkdir $root_dir/prediction first and rename the pkl to pred_results.pkl)

To visualize the results, please run
```
sh script/visualize_prediction.sh
```
Run this script will visualize all the pred results stored in model/evaluate_results. To avoid that, please move other pkl files to other directory and keep only estimator_demo_youtube_example_pose_shape_best.pkl  

The result images will be stored in model/evaluate_results/images/demo_youtube_example_pose_shape_best/


### Temporal refinment for bad samples
We use copy and paste strategy to replace bad samples (the sample with low openpose score) with prevous good sampels (the one with higher openpose score). To do that, please run
```
sh script/temporal_refine.sh
```