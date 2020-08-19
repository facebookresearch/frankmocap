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
        openpose_path,
        pred_hand_info,
    ):
        # name of the sequence
        self.seq_name = seq_name

        # sample id in terms of this sequence
        # therefore this sample id might not be the same as frame id
        self.sample_id = sample_id

        # image name
        self.img_name = img_name

        # path of the original frame
        self.frame_path = frame_path

        # openpose_path
        self.openpose_path = openpose_path

        # pred hand pose, contains both left and right hand
        self.pred_hand_info = pred_hand_info
        """
        dict{
            left_hand = dcit{
                pred_cam,
                pred_shape
                pred_pose,
                openpose_score,
                bbox
            }
            right_hand = dict{
                ...
            }
        }
        """
        
        # other information that will be updated later
        self.updated = False

        self.hand_updated = dict(
            left_hand = False,
            right_hand = False,
        )

    def update_hand(self, hand_type):
        self.hand_updated[hand_type] = True
        self.updated = True


