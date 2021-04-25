# Original code from SPIN: https://github.com/nkolot/SPIN


import torch
import numpy as np
import smplx
from smplx import SMPL as _SMPL
from smplx import SMPLX as _SMPLX
# from bodymocap.models.body_models import SMPLX as _SMPLX        #Use our custom SMPLX 
# from smplx.body_models import ModelOutput
# from bodymocap.models.body_models import ModelOutput
from smplx.lbs import vertices2joints

from bodymocap import constants

from collections import namedtuple
ModelOutput = namedtuple('ModelOutput',
                         ['vertices', 'joints', 'full_pose', 'betas',
                          'global_orient',
                          'body_pose', 'expression',
                          'left_hand_pose', 'right_hand_pose',
                          'right_hand_joints', 'left_hand_joints',
                          'jaw_pose'])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        JOINT_REGRESSOR_TRAIN_EXTRA = 'extra_data/body_module/data_from_spin//J_regressor_extra.npy'
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)        #Additional 9 joints #Check doc/J_regressor_extra.png
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)               #[N, 24 + 21, 3]  + [N, 9, 3]
        joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output



class SMPLX(_SMPLX):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        kwargs['ext'] = 'pkl'       #We have pkl file
        super(SMPLX, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        JOINT_REGRESSOR_TRAIN_EXTRA_SMPLX = 'extra_data/body_module/J_regressor_extra_smplx.npy'
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA_SMPLX)           #(9, 10475)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True

        #if pose parameter is for SMPL with 21 joints (ignoring root)
        if(kwargs['body_pose'].shape[1]==69):
            kwargs['body_pose'] = kwargs['body_pose'][:,:-2*3]        #Ignore the last two joints (which are on the palm. Not used)

        if(kwargs['body_pose'].shape[1]==23):
            kwargs['body_pose'] = kwargs['body_pose'][:,:-2]        #Ignore the last two joints (which are on the palm. Not used)

        smpl_output = super(SMPLX, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        # extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices[:,:6890])   *0      #TODO: implement this correctly

        #SMPL-X Joint order: https://docs.google.com/spreadsheets/d/1_1dLdaX-sbMkCKr_JzJW_RZCpwBwd7rcKkWT_VgAQ_0/edit#gid=0
        smplx_to_smpl = list(range(0,22)) + [28,43] + list(range(55,76)) # 28 left middle finger , 43: right middle finger 1
        smpl_joints = smpl_output.joints[:,smplx_to_smpl,:] # Convert SMPL-X to SMPL     127 ->45
        joints = torch.cat([smpl_joints, extra_joints], dim=1) # [N, 127, 3]->[N, 45, 3]  + [N, 9, 3]  # SMPL-X has more joints. should convert 45
        joints = joints[:, self.joint_map, :]     

        # Hand joints
        smplx_hand_to_panoptic = [0, 13,14,15,16, 1,2,3,17, 4,5,6,18, 10,11,12,19, 7,8,9,20] #Wrist Thumb to Pinky

        smplx_lhand =  [20] + list(range(25,40)) + list(range(66, 71))         #20 for left wrist. 20 finger joints
        lhand_joints = smpl_output.joints[:,smplx_lhand, :]      #(N,21,3)
        lhand_joints = lhand_joints[:, smplx_hand_to_panoptic, :]     #Convert SMPL-X hand order to paonptic hand order

        smplx_rhand = [21] + list(range(40,55)) + list(range(71, 76))     #21 for right wrist. 20 finger joints
        rhand_joints = smpl_output.joints[:, smplx_rhand, :]      #(N,21,3)
        rhand_joints = rhand_joints[:,smplx_hand_to_panoptic,:] #Convert SMPL-X hand order to paonptic hand order

        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             right_hand_joints=rhand_joints,     #N,21,3
                             left_hand_joints=lhand_joints,         #N,21,3
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output


"""
0	pelvis',
1	left_hip',
2	right_hip',
3	spine1',
4	left_knee',
5	right_knee',
6	spine2',
7	left_ankle',
8	right_ankle',
9	spine3',
10	left_foot',
11	right_foot',
12	neck',
13	left_collar',
14	right_collar',
15	head',
16	left_shoulder',
17	right_shoulder',
18	left_elbow',
19	right_elbow',
20	left_wrist',
21	right_wrist',
22	jaw',
23	left_eye_smplhf',
24	right_eye_smplhf',
25	left_index1',
26	left_index2',
27	left_index3',
28	left_middle1',
29	left_middle2',
30	left_middle3',
31	left_pinky1',
32	left_pinky2',
33	left_pinky3',
34	left_ring1',
35	left_ring2',
36	left_ring3',
37	left_thumb1',
38	left_thumb2',
39	left_thumb3',
40	right_index1',
41	right_index2',
42	right_index3',
43	right_middle1',
44	right_middle2',
45	right_middle3',
46	right_pinky1',
47	right_pinky2',
48	right_pinky3',
49	right_ring1',
50	right_ring2',
51	right_ring3',
52	right_thumb1',
53	right_thumb2',
54	right_thumb3',
55	nose',
56	right_eye',
57	left_eye',
58	right_ear',
59	left_ear',
60	left_big_toe',
61	left_small_toe',
62	left_heel',
63	right_big_toe',
64	right_small_toe',
65	right_heel',
66	left_thumb',
67	left_index',
68	left_middle',
69	left_ring',
70	left_pinky',
71	right_thumb',
72	right_index',
73	right_middle',
74	right_ring',
75	right_pinky',
76	right_eye_brow1',
77	right_eye_brow2',
78	right_eye_brow3',
79	right_eye_brow4',
80	right_eye_brow5',
81	left_eye_brow5',
82	left_eye_brow4',
83	left_eye_brow3',
84	left_eye_brow2',
85	left_eye_brow1',
86	nose1',
87	nose2',
88	nose3',
89	nose4',
90	right_nose_2',
91	right_nose_1',
92	nose_middle',
93	left_nose_1',
94	left_nose_2',
95	right_eye1',
96	right_eye2',
97	right_eye3',
98	right_eye4',
99	right_eye5',
100	right_eye6',
101	left_eye4',
102	left_eye3',
103	left_eye2',
104	left_eye1',
105	left_eye6',
106	left_eye5',
107	right_mouth_1',
108	right_mouth_2',
109	right_mouth_3',
110	mouth_top',
111	left_mouth_3',
112	left_mouth_2',
113	left_mouth_1',
114	left_mouth_5', # 59 in OpenPose output
115	left_mouth_4', # 58 in OpenPose output
116	mouth_bottom',
117	right_mouth_4',
118	right_mouth_5',
119	right_lip_1',
120	right_lip_2',
121	lip_top',
122	left_lip_2',
123	left_lip_1',
124	left_lip_3',
125	lip_bottom',
126	right_lip_3',
127	right_contour_1',
128	right_contour_2',
129	right_contour_3',
130	right_contour_4',
131	right_contour_5',
132	right_contour_6',
133	right_contour_7',
134	right_contour_8',
135	contour_middle',
136	left_contour_8',
137	left_contour_7',
138	left_contour_6',
139	left_contour_5',
140	left_contour_4',
141	left_contour_3',
142	left_contour_2',
143	left_contour_1'
"""


#SMPL Joints:
"""
0	pelvis',
1	left_hip',
2	right_hip',
3	spine1',
4	left_knee',
5	right_knee',
6	spine2',
7	left_ankle',
8	right_ankle',
9	spine3',
10	left_foot',
11	right_foot',
12	neck',
13	left_collar',
14	right_collar',
15	head',
16	left_shoulder',
17	right_shoulder',
18	left_elbow',
19	right_elbow',
20	left_wrist',
21	right_wrist',
22	
23	
24	nose',
25	right_eye',
26	left_eye',
27	right_ear',
28	left_ear',
29	left_big_toe',
30	left_small_toe',
31	left_heel',
32	right_big_toe',
33	right_small_toe',
34	right_heel',
35	left_thumb',
36	left_index',
37	left_middle',
38	left_ring',
39	left_pinky',
40	right_thumb',
41	right_index',
42	right_middle',
43	right_ring',
44	right_pinky',
"""