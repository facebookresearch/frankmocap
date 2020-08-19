cd model

log_dir=log
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi
train_log_dir=log/train_logs
if [ ! -d $train_log_dir ]; then
    mkdir $train_log_dir
fi
curr_date=$(date +'%m_%d_%H_%M') 
log_file="./log/train_logs/$curr_date.log"

dev=true # use dev partition or use learnfair partition
if $dev
then
    visdom_port=8097
    batch_size=256
    num_gpu=4
    partition=dev
    display_freq=1024
else
    visdom_port=8109
    echo "Use "$visdom_port" for visdom"
    batch_size=256
    num_gpu=4
    partition=learnfair
    display_freq=4096
fi
echo "We are using FAIR cluster: "$partition
port_offset=5000
dist_port=$(expr $visdom_port - $port_offset) # port for distributed training, by default, I set it as visdom_port - 5000

root_dir=/checkpoint/rongyu/data
data_root=$root_dir'/3d_hand/'
model_root=$root_dir'/models/'
blur_kernel_dir=$root_dir'/blur_kernel/'

freihand_anno_path=freihand/annotation/train.pkl
ho3d_anno_path=ho3d/annotation_tight/train.pkl
mtc_anno_path=mtc/data_processed/annotation/train.pkl
stb_anno_path=stb/annotation/train.pkl
rhd_anno_path=rhd/annotation/train.pkl
frl_anno_path=frl/annotation/all.pkl
ganerated_anno_path=ganerated/GANerated/data_processed/annotation/train.pkl
pmhand_anno_path=panoptic_hand/annotation/train.pkl

# definition of finger tips
top_joints_type=ave
pretrained_weights=$model_root'pretrained_weights/h3dw_iter_refine/joints_21_'$top_joints_type'_top_augment_all_mtc_stb_rhd.pth'

# srun --gres=gpu:$num_gpu --partition=$partition --time=4320 --cpus-per-task=10 
# srun --gpus=$num_gpu --partition=$partition --time=4320 --cpus-per-task=10 

srun --gpus-per-node=$num_gpu --partition=$partition --time=4000 --cpus-per-task 10 \
    python3 -u -m torch.distributed.launch  \
    --nproc_per_node=$num_gpu --master_port=$dist_port train.py --dist \
    --display_freq $display_freq \
    --single_branch --main_encoder resnet50 \
    --pretrained_weights $pretrained_weights \
    --batchSize $batch_size  \
    --lr_e 1e-3 --lr_decay \
    --total_epoch 200 \
    --data_root $data_root --model_root $model_root \
    --freihand_anno_path $freihand_anno_path \
    --ho3d_anno_path $ho3d_anno_path \
    --mtc_anno_path $mtc_anno_path \
    --stb_anno_path $stb_anno_path \
    --rhd_anno_path $rhd_anno_path \
    --frl_anno_path $frl_anno_path \
    --ganerated_anno_path $ganerated_anno_path \
    --pmhand_anno_path $pmhand_anno_path \
    --top_finger_joints_type $top_joints_type \
    --train_datasets freihand,ho3d,mtc,stb,rhd \
    --use_random_rescale \
    --use_random_position \
    --use_random_rotation \
    --use_color_jittering \
    --use_motion_blur \
    --blur_kernel_dir $blur_kernel_dir \
    --motion_blur_prob 0.5 \
    --sample_train_data \
    --shape_reg_weight 1.0 \
    --display_port $visdom_port  2>&1 | tee $log_file
