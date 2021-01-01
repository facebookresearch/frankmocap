# Joint Order (Position & Rotation)

## Attention !!!
For both body and hand, the order of joint position and joint angles are different. Please find the details below.

## Hand Joint
### Joint Position (Hand)

To obtain predicted 3D hand joint position, you can use [pred_joints_img](https://github.com/facebookresearch/frankmocap/blob/60584337f81795b1b9fe4f4da5ffe273f6f1266a/handmocap/hand_mocap_api.py#L222) in hand-only demo or 
[pred_lhand_joints_img](https://github.com/facebookresearch/frankmocap/blob/60584337f81795b1b9fe4f4da5ffe273f6f1266a/integration/copy_and_paste.py#L186) and [pred_rhand_joints_img](https://github.com/facebookresearch/frankmocap/blob/60584337f81795b1b9fe4f4da5ffe273f6f1266a/integration/copy_and_paste.py#L192) in body-plus-hand demo.  

The order of hand joint position is depicted as below:
<p>
    <img src="https://penincillin.github.io/frank_mocap/video_02.gif" height="200">
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
If the dimension of ```hand_pose``` is 45 (15 * 3), then the joint starts from ```Index_00```; otherwise the dimension should be 48 (16 * 3) and the joint start from wrist (or say, hand global orientation).  

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


## License
- [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file.