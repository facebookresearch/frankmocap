cd model

log_dir=log
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

test_log_dir=log/test_logs
if [ ! -d $test_log_dir ]; then
    mkdir $test_log_dir
fi

batch_size=512

root_dir=/checkpoint/rongyu/data
data_root=$root_dir'/3d_hand/'
model_root=$root_dir'/models/'

# demo_img_dir=coco/image_hand_dp/val
# demo_img_dir=demo_data/youtube_hand/image_hand
demo_img_dir=demo_data/youtube_example/image_hand

# test_checkpoint_dir=checkpoints_best
test_checkpoint_dir=checkpoints_best

test_dataset=demo
log_file="./log/test_logs/test_log_$test_dataset.log"

# srun --gpus-per-node=1 --partition=dev --time=4000 --cpus-per-task 10 python3 test.py \
CUDA_VISIBLE_DEVICES=1 python3 demo.py \
    --single_branch --main_encoder resnet50 \
    --data_root $data_root --model_root $model_root \
    --demo_img_dir $demo_img_dir \
    --batchSize $batch_size --phase test \
    --test_dataset $test_dataset \
    --test_checkpoint_dir $test_checkpoint_dir \
    2>&1 | tee $log_file 
