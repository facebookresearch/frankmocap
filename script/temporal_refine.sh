cd h3dw_util

root_dir=/checkpoint/rongyu/data/3d_hand/demo_data/youtube_example

# apply copy and paste refinement
apply_copy_paste=1

# apply average frame refinement
apply_average_frame=0

# visualize the results
visualize=1

# if you want to visualize updated sample, set $updated_only=1, otherwise 0
updated_only=1

# apply temporal refine
python src/demo/temporal_two_hands/main.py $root_dir $apply_copy_paste $apply_average_frame $visualize $updated_only