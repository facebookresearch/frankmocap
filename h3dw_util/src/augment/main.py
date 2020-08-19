import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
import ry_utils
import parallel_io as pio
import copy
import multiprocessing as mp

from augment.sample import Sample
from augment.data_loader import load_all_samples
from visualize import visualize_body, visualize_hand
import config
from augment.augment_model import AugmentModel


def get_all_samples():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace"
    exp_res_dir = osp.join(root_dir, "data/coco/prediction")
    '''
    res_hand_data_file = osp.join(root_dir, "data/coco/prediction/hand_info.pkl")
    all_samples = load_all_samples(exp_res_dir, res_hand_data_file)
    '''
    all_samples = load_all_samples(exp_res_dir)
    return all_samples


def visualize_samples(all_samples):
    res_dir = "visualization/augment/render_hand/coco"
    visualize_hand(all_samples, res_dir)

    res_dir = "visualization/augment/render_body/coco"
    visualize_body(all_samples, res_dir, updated_only=False, scale_ratio=1)


def apply_update_wrist(all_samples):
    strategy = 'update_wrist'
    for seq_name in all_samples.keys():
        samples = all_samples[seq_name]
        t_model = AugmentModel(config, samples, strategy)
        t_model.update_sample()


def write_to_file(all_samples):
    all_data = dict()
    for seq_name in all_samples.keys():
        for sample in all_samples[seq_name]:
            img_id = sample.img_name
            print(img_id)
            data = dict(
                img_id = img_id,
                left_hand_exist = False,
                left_hand_natural = False,
                right_hand_exist = False,
                right_hand_natural = False,
                left_hand_wrist_local = np.zeros((3,)),
                right_hand_wrist_local = np.zeros((3,)),
                left_hand_wrist_global = np.zeros((3,)),
                right_hand_wrist_global = np.zeros((3,)),
                left_hand_cam = np.zeros((3,)),
                right_hand_cam = np.zeros((3,)),
                left_hand_pose = np.zeros((45,)),
                right_hand_pose = np.zeros((45,)),
                left_hand_verts = np.zeros((778, 3), dtype=np.float16),
                right_hand_verts = np.zeros((778, 3), dtype=np.float16),
                left_hand_score = -1,
                right_hand_score = -1,
            )
            has_hand = False
            for hand_type in ['left_hand', 'right_hand']:
                if len(sample.hand_img_path[hand_type])>0:
                    has_hand = True
                    key = f"{hand_type}_exist"
                    data[key] = True
                    if sample.hand_valid[hand_type]:
                        key = f"{hand_type}_natural"
                        data[key] = True
                    key = f"{hand_type}_wrist_local"
                    data[key][:] = sample.pred_hand_info[hand_type]['wrist_rot_local'][:]
                    key = f"{hand_type}_wrist_global"
                    data[key][:] = sample.pred_hand_info[hand_type]['pred_hand_pose'][:3]
                    key = f"{hand_type}_cam"
                    data[key][:] = sample.pred_hand_info[hand_type]['pred_hand_cam'][:]
                    key = f"{hand_type}_pose"
                    data[key][:] = sample.pred_hand_info[hand_type]['pred_hand_pose'][3:]
                    key = f"{hand_type}_verts"
                    data[key][:] = sample.pred_hand_info[hand_type]['pred_hand_verts'][:].astype(np.float16)
                    key = f"{hand_type}_score"
                    data[key] = sample.pose_prior_score[hand_type]
            if has_hand:
                all_data[img_id] = data
    
    res_data_file = "/Users/rongyu/Documents/research/FAIR/workplace/data/coco/prediction/hand_info.pkl"
    pio.save_pkl_single(res_data_file, all_data)
    
    # check data
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/coco/prediction"
    log_file = osp.join(root_dir, "augment_log.txt")
    with open(log_file, "w") as out_f:
        for data in all_data.values():
            for hand_type in ['left_hand', 'right_hand']:
                if not data[f'{hand_type}_exist']:
                    assert not data[f'{hand_type}_natural']
                if data[f'{hand_type}_exist']:
                    assert np.all(np.abs(data[f'{hand_type}_pose']) > 1e-10), data[f'{hand_type}_pose']
                if data[f'{hand_type}_exist'] and not data[f'{hand_type}_natural']:
                    print(data['img_id'], hand_type, data[f'{hand_type}_score'])
                    out_f.write(f"{data['img_id']}, {hand_type}, {data[f'{hand_type}_score']}\n")


def main():
    '''
    all_samples = get_all_samples()
    pio.save_pkl_single('data/augment/all_samples.pkl', all_samples)
    '''
    all_samples = pio.load_pkl_single('data/augment/all_samples.pkl')
    
    pick_samples = dict()
    pick_samples['train'] = list()
    samples = pick_samples['train']
    for sample in all_samples['train']:
        if sample.img_name.find("COCO_train2014_000000000625")>=0:
            samples.append(sample)
    visualize_samples(pick_samples)
    sys.exit(0)
    # sys.exit(0)

    apply_update_wrist(all_samples)

    write_to_file(all_samples)
    
if __name__ == '__main__':
    main()