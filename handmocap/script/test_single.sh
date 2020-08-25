cd model

log_dir=log
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

test_log_dir=log/test_logs
if [ ! -d $test_log_dir ]; then
    mkdir $test_log_dir
fi

# batch_size=512
batch_size=512

root_dir=/checkpoint/rongyu/data
data_root=$root_dir'/3d_hand/'
model_root=$root_dir'/models/'

freihand_anno_path=freihand/annotation/val.pkl
ho3d_anno_path=ho3d/annotation_tight/val.pkl
stb_anno_path=stb/annotation/val.pkl
rhd_anno_path=rhd/annotation/val.pkl
demo_img_dir=demo_data/youtube/origin/image_hand
pmhand_anno_path=panoptic_hand/annotation/val.pkl # pmhand is panoptic-mpii hand

# test_checkpoint_dir=checkpoints
test_checkpoint_dir=checkpoints_best

test_dataset=pmhand # pmhand is panoptic-mpii hand
log_file="./log/test_logs/test_log_$test_dataset.log"
# srun --gpus-per-node=1 --partition=dev --time=4000 --cpus-per-task 10 python3 test.py \
CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --single_branch --main_encoder resnet50 \
    --data_root $data_root --model_root $model_root \
    --freihand_anno_path $freihand_anno_path \
    --ho3d_anno_path $ho3d_anno_path \
    --stb_anno_path $stb_anno_path \
    --rhd_anno_path $rhd_anno_path \
    --pmhand_anno_path $pmhand_anno_path \
    --demo_img_dir $demo_img_dir \
    --batchSize $batch_size --phase test \
    --test_dataset $test_dataset \
    --test_checkpoint_dir $test_checkpoint_dir \
    2>&1 | tee $log_file