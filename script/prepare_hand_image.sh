cd h3dw_util

root_dir=/checkpoint/rongyu/data/3d_hand/demo_data/youtube_example

# visualize output of openpose
python src/demo/single_person/visualize_openpose_output.py $root_dir

# crop hand
python src/demo/single_person/crop_hand.py $root_dir