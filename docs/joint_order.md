# Joint Order (Position & Rotation)

## Attention !!!
The orders of joint position and joint angle are different. The details are listed below.

## Hand Joint
### Joint Position (Hand)

The joint positions are  converted to image space (X,Y coordinates are aligned to image, Z coordinates are rescaled accordingly.)  

To obtain predicted 3D hand joint position, you can use [pred_joints_img](https://github.com/facebookresearch/frankmocap/blob/60584337f81795b1b9fe4f4da5ffe273f6f1266a/handmocap/hand_mocap_api.py#L222) in hand-only demo or 
[pred_lhand_joints_img](https://github.com/facebookresearch/frankmocap/blob/60584337f81795b1b9fe4f4da5ffe273f6f1266a/integration/copy_and_paste.py#L186) and [pred_rhand_joints_img](https://github.com/facebookresearch/frankmocap/blob/60584337f81795b1b9fe4f4da5ffe273f6f1266a/integration/copy_and_paste.py#L192) in body-plus-hand demo.  

The order of hand joint position is visualized below:

<p>
    <img src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/.github/media/keypoints_hand.png" height="500">
</p>


The order of hand joint (position) is listed below:
```
0 : Wrist
1 : Thumb_00
2 : Thumb_01
3 : Thumb_02
4 : Thumb_03
5 : Index_00
6 : Index_01
7 : Index_02
8 : Index_03
9 : Middle_00
10 : Middle_01
11 : Middle_02
12 : Middle_03
13 : Ring_00
14 : Ring_01
15 : Ring_02
16 : Ring_03
17 : Little_00
18 : Little_01
19 : Little_02
20 : Little_03
```

### Joint Angle (Hand)
To obtain predicted 3D hand joint angles (in [angle-axis format](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)), you can use [pred_hand_pose](https://github.com/facebookresearch/frankmocap/blob/60584337f81795b1b9fe4f4da5ffe273f6f1266a/handmocap/hand_mocap_api.py#L197) in hand-only demo or [pred_left_hand_pose](https://github.com/facebookresearch/frankmocap/blob/60584337f81795b1b9fe4f4da5ffe273f6f1266a/integration/copy_and_paste.py#L234) [pred_right_hand_pose](https://github.com/facebookresearch/frankmocap/blob/60584337f81795b1b9fe4f4da5ffe273f6f1266a/integration/copy_and_paste.py#L235) in body-plus-hand demo.  

The axis of joint angle is depicted below (right-hand rule):
<p>
    <img src="https://penincillin.github.io/project/frankmocap_iccvw2021/axis.png" height="300">
</p>


If the dimension of ```hand_pose``` is 45 (15 * 3), then the joint starts from ```Index_00```; otherwise the dimension should be 48 (16 * 3) and the joint starts from wrist (or say, hand global orientation).  

The order of hand joint (angle) is listed below:
```
0 : Wrist
1 : Index_00
2 : Index_01
3 : Index_02
4 : Middle_00
5 : Middle_01
6 : Middle_02
7 : Little_00
8 : Little_01
9 : Little_02
10 : Ring_00
11 : Ring_01
12 : Ring_02
13 : Thumb_00
14 : Thumb_01
15 : Thumb_02
```


## Body Joint
### Joint Position (Body)

The joint positions are  converted to image space (X,Y coordinates are aligned to image, Z coordinates are rescaled accordingly.)  

To obtain predicted 3D body joint position, you can use [pred_joints_img](https://github.com/facebookresearch/frankmocap/blob/44f4f6718a45baf0836c9785f02ea1d74f6f5774/bodymocap/body_mocap_api.py#L112) in body-only demo or 
[pred_body_joints_img](https://github.com/facebookresearch/frankmocap/blob/60584337f81795b1b9fe4f4da5ffe273f6f1266a/integration/copy_and_paste.py#L179) in body-plus-hand demo.  

The order of body joint (position) is listed below:
```
0: OP_Nose
1: OP_Neck
2: OP_R_Shoulder
3: OP_R_Elblow
4: OP_R_Wrist
5: OP_L_Shoulder
6: OP_L_Elbow
7: OP_L_Wrist
8: OP_Middle_Hip
9: OP_R_Hip
10: OP_R_Knee
11: OP_R_Ankle
12: OP_L_Hip
13: OP_L_Knee
14: OP_L_Ankle
15: OP_R_Eye
16: OP_L_Eye
17: OP R_Ear
18: OP_L_Ear
19: OP_L_Big_Toe
20: OP_L_Small_Toe
21: OP_L_Heel
22: OP_R_Big_Toe
23: OP_R_Small_Toe
24: OP_R_Heel
25: R_Ankle
26: R_Knee
27: R_Hip
28: L_Hip
29: L_Knee
30: L_Ankle
31: R_Wrist
32: R_Elbow
33: R_Shoulder
34: L_Shoulder
35: L_Elbow
36: L_Wrist
37: Neck (LSP)
38: Top of Head (LSP)
39: Pelvis (MPII)
40: Thorax (MPII)
41: Spine (H36M)
42: Jaw (H36M)
43: Head (H36M)
44: Nose
45: L_Eye
46: R_Eye
47: L_Ear
48: R_Ear
```

### Joint Angle (Body)
To obtain predicted 3D body joint angles (in [angle-axis format](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)), you can use [pred_body_pose](https://github.com/facebookresearch/frankmocap/blob/44f4f6718a45baf0836c9785f02ea1d74f6f5774/bodymocap/body_mocap_api.py#L115) in body-only demo or [pred_body_pose](https://github.com/facebookresearch/frankmocap/blob/60584337f81795b1b9fe4f4da5ffe273f6f1266a/integration/copy_and_paste.py#L164) in body-plus-hand demo.  

The dimesion should be 72 (24 * 3).  It is worth noting that if SMPL-X is used for body module, then the 22-th and 23-th body joint angles are invalid, we keep it for the consistent format with SMPL.

The order of body joint (angle) is listed below:
```
0: Global
1: L_Hip
2: R_Hip
3: Spine_01
4: L_Knee
5: R_Knee
6: Spine_02
7: L_Ankle
8: R_Ankle
9: Spine_03
10: L_Toe
11: R_Toe
12: Neck
13: L_Collar
14: R_Collar
15: Head
16: L_Shoulder
17: R_Shoulder
18: L_Elbow
19: R_Elbow
20: L_Wrist
21: R_Wrist
22: L_Palm (Invalid for SMPL-X)
23: R_Palm (Invalid for SMPL-X)
```

The skeleton of SMPL body is depicted below, for SMPL-X body, the 22-th and 23-th body joint are invalid:
<p>
    <img src="https://penincillin.github.io/project/frankmocap_iccvw2021/body_skeleton.png" height="500">
</p>



## License
- [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file.