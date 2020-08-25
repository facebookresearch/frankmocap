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
        openpose_score,
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
        
        # openpose score on hand, contains both left and right hand
        self.openpose_score = openpose_score

        # other information that will be updated later
        self.updated = False

        self.hand_updated = dict(
            left_hand = False,
            right_hand = False,
        )

        self.select_sample_render_path = dict(
            left_hand = '',
            right_hand = ''
        )
    
    def update_hand(self, hand_type, select_render_img_path=''):
        self.hand_updated[hand_type] = True
        self.select_sample_render_path[hand_type] = select_render_img_path


