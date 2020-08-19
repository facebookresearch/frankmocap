import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
import ry_utils
import parallel_io as pio

from temporal.sample import Sample
from temporal.data_loader import load_all_samples
from visualize import visualize_body, visualize_hand
import copy
from temporal.temporal_model import TemporalModel
import multiprocessing as mp


def get_all_samples():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace"
    data_dir = osp.join(root_dir, "data/demo_data/body_capture")
    exp_res_dir = osp.join(root_dir, 
        "experiment/experiment_results/3d_hand/h3dw/demo_data/body_capture/prediction")
    all_samples = load_all_samples(data_dir, exp_res_dir)
    return all_samples


def visualize_samples(all_samples):
    '''
    res_dir = "visualization/temporal/render_hand/origin"
    visualize_hand(all_samples, res_dir)

    res_dir = "visualization/temporal/render_body/origin/hand_wrist_rot"
    visualize_body(all_samples, res_dir, updated_only=False, scale_ratio=2)
    '''


def apply_copy_and_paste(all_samples):
    import temporal.config as config
    all_samples_new = dict()
    strategy = 'copy_and_paste'
    for seq_name in all_samples.keys():
        samples = all_samples[seq_name]
        t_model = TemporalModel(config, samples, strategy)
        samples_new = t_model.update_sample()
        all_samples_new[seq_name] = samples_new
    sys.stdout.flush()

    res_dir = "visualization/temporal/render_body/updated_use_wrist_rot"
    visualize_body(all_samples_new, res_dir, updated_only=True, scale_ratio=2)


def apply_update_wrist(all_samples):
    import temporal.config as config
    all_samples_new = dict()
    strategy = 'update_wrist'
    for seq_name in all_samples.keys():
        samples = all_samples[seq_name]
        t_model = TemporalModel(config, samples, strategy)
        samples_new = t_model.update_sample()
        all_samples_new[seq_name] = samples_new
        num_updated = 0
        # all_samples_new[seq_name] = samples

    res_dir = "visualization/temporal/render_body/update_wrist/updated"
    visualize_body(all_samples_new, res_dir, updated_only=True, scale_ratio=2)


def apply_average_frame_single(update_wrist, win_size, all_samples):
    import temporal.config as config
    config.strategy_params['average_frame']['win_size'] = win_size
    if update_wrist:
        all_samples_new = dict()
        strategy = 'update_wrist'
        for seq_name in all_samples.keys():
            samples = all_samples[seq_name]
            t_model = TemporalModel(config, samples, strategy)
            samples_new = t_model.update_sample()
            all_samples_new[seq_name] = samples_new
        all_samples = all_samples_new

    strategy = 'average_frame'
    all_samples_new = dict()
    for seq_name in all_samples.keys():
        samples = all_samples[seq_name]
        t_model = TemporalModel(config, samples, strategy)
        samples_new = t_model.update_sample()
        all_samples_new[seq_name] = samples_new

    if update_wrist:
        res_name = f"{win_size}_wrist_combine"
    else:
        res_name = f"{win_size}_wrist_hand"

    res_dir = osp.join("visualization/temporal/render_body/average", res_name)
    visualize_body(all_samples_new, res_dir, updated_only=True, scale_ratio=2)
    print(f"{res_name} complete")

def apply_average_frame(all_samples):
    p_list = list()
    for win_size in [3, 5]:
        for update_wrist in [1, 0]:
            apply_average_frame_single(update_wrist, win_size, all_samples)


def main():
    '''
    all_samples = get_all_samples()
    pio.save_pkl_single('data/temporal/all_samples.pkl', all_samples)
    '''
    all_samples = pio.load_pkl_single('data/temporal/all_samples.pkl')
    apply_average_frame(all_samples)

    
if __name__ == '__main__':
    main()