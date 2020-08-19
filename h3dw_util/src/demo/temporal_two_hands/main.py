import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
import ry_utils
import parallel_io as pio
import copy
import multiprocessing as mp
import cv2

from demo.temporal_two_hands.sample import Sample
from demo.temporal_two_hands.data_loader import load_all_samples
from demo.temporal_two_hands.visualize import visualize_two_hands
from demo.temporal_two_hands.temporal_model import TemporalModel


def visualize_samples(all_samples, res_dir, updated_only=False, render_type='direct'):
    assert render_type in ['direct', 'multi']

    all_samples_list = list()
    for seq_name in all_samples:
        all_samples_list += all_samples[seq_name]

    # make subdirs
    for sample in all_samples_list:
        seq_name = sample.frame_path.split('/')[-2]
        res_subdir = osp.join(res_dir, seq_name)
        ry_utils.build_dir(res_subdir)

    if render_type == 'direct':
        visualize_two_hands(all_samples_list, res_dir, updated_only=updated_only)
    
    else:
        num_process = 16
        num_data = len(all_samples_list)
        num_each = num_data // num_process
        res_sh_file = "run_render.sh"

        split_sample_dir = "data/split_samples"
        ry_utils.build_dir(split_sample_dir)
        ry_utils.build_dir("data/render_log")

        with open(res_sh_file, "w") as out_f:
            for i in range(num_process):
                start = i*num_each
                end = (i+1)*num_each if i<num_process-1 else num_data
                samples = all_samples_list[start:end]
                res_pkl_file = osp.join(split_sample_dir, f"split_{i:02d}.pkl")
                pio.save_pkl_single(res_pkl_file, samples)

                line = f"python src/demo/temporal_two_hands/visualize.py {res_pkl_file} {res_dir} {int(updated_only)} " + \
                    f" 2>&1 | tee data/render_log/{i:02d}.log & \n"
                out_f.write(line)
        
        os.system("sh run_render.sh")
        # print("Please run run_render.sh to do rendering")


def save_data(all_samples, out_dir):
    pred_results = dict()
    # for sample in all_samples_list:
    for seq_name in all_samples:
        samples = all_samples[seq_name]

        for sample in samples:
            img_name = sample.img_name
            pred_results[img_name] = sample.pred_hand_info
    
    res_pred_file = osp.join(out_dir, "pred_hand_info.pkl")
    ry_utils.make_subdir(res_pred_file)
    pio.save_pkl_single(res_pred_file, pred_results)


def apply_copy_and_paste(all_samples):
    import demo.temporal_two_hands.config as config
    all_samples_new = dict()
    strategy = 'copy_and_paste'
    for seq_name in all_samples.keys():
        samples = all_samples[seq_name]
        t_model = TemporalModel(config, samples, strategy)
        samples_new = t_model.update_sample()
        all_samples_new[seq_name] = samples_new
        # print(len(samples), len(samples_new))
    sys.stdout.flush()
    return all_samples_new
   

def apply_average_frame_func(all_samples, win_size=3):
    import demo.temporal_two_hands.config as config
    config.strategy_params['average_frame']['win_size'] = win_size

    strategy = 'average_frame'
    all_samples_new = dict()
    for seq_name in all_samples.keys():
        samples = all_samples[seq_name]
        t_model = TemporalModel(config, samples, strategy)
        samples_new = t_model.update_sample()
        all_samples_new[seq_name] = samples_new
    return all_samples_new


def main():
    root_dir = sys.argv[1]
    apply_copy_paste=int(sys.argv[2])
    apply_average_frame=int(sys.argv[3])
    visualize=int(sys.argv[4])
    updated_only=int(sys.argv[5])

    # load all samples first
    all_samples = load_all_samples(root_dir) # dict

    if apply_copy_paste:
        print("Apply copy paste")
        all_samples_new = apply_copy_and_paste(all_samples)
    else:
        all_samples_new = all_samples
    
    if apply_average_frame:
        print("Apply average frame")
        # win_size means considering how many adjacent frames each time
        all_samples_new = apply_average_frame_func(all_samples, win_size=3)

    # save the results to $root_dir/pred_hand_info.pkl
    save_data(all_samples_new, root_dir)

    if visualize:
        # visualize the results to original frames, if you don't want that part, please annotate the following code
        res_dir = osp.join(root_dir, "prediction/origin_frame_visualize")
        # if set updated_only = True, then it will only visualize updated samples (by copy and paste strategy) otherwise, it will visualize all the samples
        # render_type = 'direct', it will using single process to render and it will be slow, change to it multi, then it will use multiple process to render
        visualize_samples(all_samples_new, res_dir, updated_only=updated_only, render_type='multi')


if __name__ == '__main__':
    main()