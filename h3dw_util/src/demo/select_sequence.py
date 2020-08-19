import os, sys, shutil
import os.path as osp
import ry_utils
import parallel_io as pio


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/"

    select_info = dict(
        youtube_processed_01 = ("cello", "oliver_01"),
        youtube_processed_02 = ("cook_02_01", "cook_02_03", 'legao_02_01'),
        youtube_processed_03 = ("lecture_04_02", "lecture_01_01")
    )

    res_dir = osp.join(root_dir, "youtube/origin")
    ry_utils.build_dir(res_dir)

    pred_results_new = list()
    for in_dir in select_info:

        # frame
        for seq_name in select_info[in_dir]:
            ry_utils.build_dir(osp.join(root_dir, res_dir, "frame"))
            frame_in_dir = osp.join(root_dir, in_dir, "frame", seq_name)
            frame_out_dir = osp.join(root_dir, res_dir, "frame", seq_name)
            if osp.exists(frame_out_dir): continue
            shutil.copytree(frame_in_dir, frame_out_dir)
        
        # bbox_info
        for seq_name in select_info[in_dir]:
            for hand_type in ['left', 'right']:
                ry_utils.build_dir(osp.join(root_dir, res_dir, "bbox_info"))
                bbox_info_in_file = osp.join(root_dir, in_dir, "bbox_info", f"{seq_name}_{hand_type}_bbox.pkl")
                bbox_info_out_file = osp.join(root_dir, res_dir, "bbox_info", f"{seq_name}_{hand_type}_bbox.pkl")
                shutil.copy2(bbox_info_in_file, bbox_info_out_file)
        

        # openpose_output
        for seq_name in select_info[in_dir]:
            ry_utils.build_dir(osp.join(root_dir, res_dir, "openpose_output"))
            ov_in_dir = osp.join(root_dir, in_dir, "openpose_output", seq_name)
            ov_out_dir = osp.join(root_dir, res_dir, "openpose_output", seq_name)
            # if osp.exists(ov_out_dir): continue
            shutil.copytree(ov_in_dir, ov_out_dir)
        
        # openpose visualization
        for seq_name in select_info[in_dir]:
            ry_utils.build_dir(osp.join(root_dir, res_dir, "openpose_visualization"))
            ov_in_dir = osp.join(root_dir, in_dir, "openpose_visualization", seq_name)
            ov_out_dir = osp.join(root_dir, res_dir, "openpose_visualization", seq_name)
            if osp.exists(ov_out_dir): continue
            shutil.copytree(ov_in_dir, ov_out_dir)
        
        '''
        # rendered_frames
        for seq_name in select_info[in_dir]:
            ry_utils.build_dir(osp.join(root_dir, res_dir, "prediction/h3dw/origin_frame"))
            frame_in_dir = osp.join(root_dir, in_dir, "prediction/h3dw/origin_frame", seq_name)
            frame_out_dir = osp.join(root_dir, res_dir, "prediction/h3dw/origin_frame", seq_name)
            if osp.exists(frame_out_dir): continue
            shutil.copytree(frame_in_dir, frame_out_dir)

        # rendered hand
        for seq_name in select_info[in_dir]:
            for hand_type in ['left', 'right']:
                ry_utils.build_dir(osp.join(root_dir, res_dir, f"prediction/h3dw/224_size/{hand_type}_hand"))
                frame_in_dir = osp.join(root_dir, in_dir, f"prediction/h3dw/224_size/{hand_type}_hand", seq_name)
                frame_out_dir = osp.join(root_dir, res_dir, f"prediction/h3dw/224_size/{hand_type}_hand", seq_name)
                if osp.exists(frame_out_dir): continue
                shutil.copytree(frame_in_dir, frame_out_dir)
        '''

        # cropped hand
        for seq_name in select_info[in_dir]:
            for hand_type in ['left', 'right']:
                ry_utils.build_dir(osp.join(root_dir, res_dir, f"image_hand/{hand_type}_hand"))
                frame_in_dir = osp.join(root_dir, in_dir, f"image_hand/{hand_type}_hand", seq_name)
                frame_out_dir = osp.join(root_dir, res_dir, f"image_hand/{hand_type}_hand", seq_name)
                if osp.exists(frame_out_dir): continue
                shutil.copytree(frame_in_dir, frame_out_dir)


        # pred results
        pred_results_file = osp.join(root_dir, in_dir, "prediction/h3dw/pred_results_youtube.pkl")
        pred_results = pio.load_pkl_single(pred_results_file)
        for single_data in pred_results:
            img_name = single_data['img_name']
            seq_name_pred = img_name.split('/')[-2]
            if seq_name_pred in select_info[in_dir]:
                pred_results_new.append(single_data)
        
        print(in_dir, "completes")
        
    
    res_pred_file = osp.join(root_dir, "youtube", "prediction/h3dw/pred_results_youtube.pkl")
    ry_utils.make_subdir(res_pred_file)
    pio.save_pkl_single(res_pred_file, pred_results_new)


if __name__ == '__main__':
    main()