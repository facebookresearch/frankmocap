import os, sys, shutil
sys.path.append("src")
import os.path as osp
import parallel_io as pio
import utils.geometry_utils as gu
import torch

def get_smplx_pose(pose_rotmat):
    pose_rotmat = torch.from_numpy(pose_rotmat).float()
    pad = torch.ones((72,)).float().view(24, 3, 1)
    pose_rotmat = torch.cat((pose_rotmat, pad), axis=2)
    global_rot_rotmat = pose_rotmat[0:1, :, :]

    global_rot = gu.rotation_matrix_to_angle_axis(global_rot_rotmat)
    smplx_pose = gu.rotation_matrix_to_angle_axis(pose_rotmat[1:22, :, :]).view(1, 21, 3)
    return global_rot, smplx_pose


def load_eft_fitting(eft_fitting_dir):
    fitting_results = dict()
    num_file = 0
    for subdir, dirs, files in os.walk(eft_fitting_dir):
        for file in files:
            if file.endswith(".pkl"):
                file_path = osp.join(subdir, file)
                all_data = pio.load_pkl_single(file_path)
                for data in all_data.values():
                    assert len(data['imageName']) == 1
                    img_path = data['imageName'][0]
                    img_name = '/'.join(img_path.split('/')[-3:])

                    pred_pose_rotmat = data['pred_pose_rotmat'][0]
                    global_rot, smplx_pose = get_smplx_pose(pred_pose_rotmat)
                    fitting_results[img_name] = dict(
                        global_rot = global_rot, 
                        smplx_pose = smplx_pose,
                        smplx_shape = data['pred_shape'])
                print(file_path)
    return fitting_results


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/mtc/data_original/eft_fitting"

    '''
    eft_fitting_dir = osp.join(root_dir, "04-24_panoptic_with8143_iter80_hand")
    fitting_results = load_eft_fitting(eft_fitting_dir)
    res_file = osp.join(root_dir, "eft_fitting_two_hands.pkl")
    pio.save_pkl_single(res_file, fitting_results)
    '''

    valid_seq_name = "171026_pose1" 
    valid_frame_str = [f"{id:08d}" for id in (
        2975, 3310, 13225, 14025, 16785)]

    select_data = dict()
    in_file = osp.join(root_dir, "eft_fitting_two_hands.pkl")
    all_data = pio.load_pkl_single(in_file)
    for img_name in all_data:
        record = img_name.split('/')
        if record[0] == valid_seq_name:
            if record[1] in valid_frame_str:
                select_data[img_name] = all_data[img_name]
    res_file = osp.join(root_dir, "eft_fitting_two_hands_selected.pkl")
    pio.save_pkl_single(res_file, select_data)


if __name__ == "__main__":
    main()