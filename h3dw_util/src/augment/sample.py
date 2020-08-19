import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np

class Sample(object):

    def __init__(self, 
        seq_name,
        sample_id,
        img_name,
        frame_path,
        hand_img_path,
        render_img_path,
        pred_body_info,
        pred_hand_info,
    ):
        # name of the sequence
        self.seq_name = seq_name

        # sample id in terms of this sequence
        # therefore this sample id might be the same as frame id
        self.sample_id = sample_id

        self.img_name = img_name

        # path of the original frame
        self.frame_path = frame_path

        # All images has been flipped to right
        self.hand_img_path = hand_img_path

        # All images has been flipped to right
        # please be remind that render image might contain 
        self.render_img_path = render_img_path

        # pred smpl body pose, (from fairmocap)
        self.pred_body_info = pred_body_info
        
        # pred hand pose, contains both left and right hand
        self.pred_hand_info = pred_hand_info

        self.hand_valid = dict(
            left_hand = False,
            right_hand = False
        )

        self.pose_prior_score = dict(
            left_hand = -1,
            right_hand = -1,
        )