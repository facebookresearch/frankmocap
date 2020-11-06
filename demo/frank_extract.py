# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import os.path as osp
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import pickle

############# input parameters  #############
from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from handmocap.hand_mocap_api import HandMocap
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer
from datetime import datetime

from bodymocap.body_bbox_detector import BodyPoseEstimator
from handmocap.hand_bbox_detector import HandBboxDetector
from intergraion.copy_and_paste import intergration_copy_paste

from renderer.viewer2D import ImShow

def generate_json_structure():
    output_json = {
      "posenet": {
          "nose": {
              "x": [],
              "y": [],
              "z": []
          },
          "leftEye": {
              "x": [],
              "y": [],
              "z": []
          },
          "rightEye": {
              "x": [],
              "y": [],
              "z": []
          },
          "leftEar": {
              "x": [],
              "y": [],
              "z": []
          },
          "rightEar": {
              "x": [],
              "y": [],
              "z": []
          },
          "leftShoulder": {
              "x": [],
              "y": [],
              "z": []
          },
          "rightShoulder": {
              "x": [],
              "y": [],
              "z": []
          },
          "leftElbow": {
              "x": [],
              "y": [],
              "z": []
          },
          "rightElbow": {
              "x": [],
              "y": [],
              "z": []
          },
          "leftWrist": {
              "x": [],
              "y": [],
              "z": []
          },
          "rightWrist": {
              "x": [],
              "y": [],
              "z": []
          },
          "leftHip": {
              "x": [],
              "y": [],
              "z": []
          },
          "rightHip": {
              "x": [],
              "y": [],
              "z": []
          },
          "leftKnee": {
              "x": [],
              "y": [],
              "z": []
          },
          "rightKnee": {
              "x": [],
              "y": [],
              "z": []
          },
          "leftAnkle": {
              "x": [],
              "y": [],
              "z": []
          },
          "rightAnkle": {
              "x": [],
              "y": [],
              "z": []
          }
      },
      "leftHand": {
          "hand": {
              "x": [],
              "y": [],
              "z": []
          },
          "thumb_root": {
              "x": [],
              "y": [],
              "z": []
          },
          "thumb_base": {
              "x": [],
              "y": [],
              "z": []
          },
          "thumb_mid": {
              "x": [],
              "y": [],
              "z": []
          },
          "thumb_tip": {
              "x": [],
              "y": [],
              "z": []
          },
          "index_root": {
              "x": [],
              "y": [],
              "z": []
          },
          "index_base": {
              "x": [],
              "y": [],
              "z": []
          },
          "index_mid": {
              "x": [],
              "y": [],
              "z": []
          },
          "index_tip": {
              "x": [],
              "y": [],
              "z": []
          },
          "middle_root": {
              "x": [],
              "y": [],
              "z": []
          },
          "middle_base": {
              "x": [],
              "y": [],
              "z": []
          },
          "middle_mid": {
              "x": [],
              "y": [],
              "z": []
          },
          "middle_tip": {
              "x": [],
              "y": [],
              "z": []
          },
          "ring_root": {
              "x": [],
              "y": [],
              "z": []
          },
          "ring_base": {
              "x": [],
              "y": [],
              "z": []
          },
          "ring_mid": {
              "x": [],
              "y": [],
              "z": []
          },
          "ring_tip": {
              "x": [],
              "y": [],
              "z": []
          },
          "pinky_root": {
              "x": [],
              "y": [],
              "z": []
          },
          "pinky_base": {
              "x": [],
              "y": [],
              "z": []
          },
          "pinky_mid": {
              "x": [],
              "y": [],
              "z": []
          },
          "pinky_tip": {
              "x": [],
              "y": [],
              "z": []
          }
      },
      "rightHand": {
          "hand": {
              "x": [],
              "y": [],
              "z": []
          },
          "thumb_root": {
              "x": [],
              "y": [],
              "z": []
          },
          "thumb_base": {
              "x": [],
              "y": [],
              "z": []
          },
          "thumb_mid": {
              "x": [],
              "y": [],
              "z": []
          },
          "thumb_tip": {
              "x": [],
              "y": [],
              "z": []
          },
          "index_root": {
              "x": [],
              "y": [],
              "z": []
          },
          "index_base": {
              "x": [],
              "y": [],
              "z": []
          },
          "index_mid": {
              "x": [],
              "y": [],
              "z": []
          },
          "index_tip": {
              "x": [],
              "y": [],
              "z": []
          },
          "middle_root": {
              "x": [],
              "y": [],
              "z": []
          },
          "middle_base": {
              "x": [],
              "y": [],
              "z": []
          },
          "middle_mid": {
              "x": [],
              "y": [],
              "z": []
          },
          "middle_tip": {
              "x": [],
              "y": [],
              "z": []
          },
          "ring_root": {
              "x": [],
              "y": [],
              "z": []
          },
          "ring_base": {
              "x": [],
              "y": [],
              "z": []
          },
          "ring_mid": {
              "x": [],
              "y": [],
              "z": []
          },
          "ring_tip": {
              "x": [],
              "y": [],
              "z": []
          },
          "pinky_root": {
              "x": [],
              "y": [],
              "z": []
          },
          "pinky_base": {
              "x": [],
              "y": [],
              "z": []
          },
          "pinky_mid": {
              "x": [],
              "y": [],
              "z": []
          },
          "pinky_tip": {
              "x": [],
              "y": [],
              "z": []
          }
      }
    }
    return output_json

def fill_body_joints(output_json,pred_output_list):
    correspondence =[["nose",0],
    ["leftEye",16],
    ["rightEye",15],
    ["leftEar",18],
    ["rightEar",17],
    ["leftShoulder",5],
    ["rightShoulder",2],
    ["leftElbow",6],
    ["rightElbow",3],
    ["leftWrist",7],
    ["rightWrist",4],
    ["leftHip",12],
    ["rightHip",9],
    ["leftKnee",13],
    ["rightKnee",10],
    ["leftAnkle",14],
    ["rightAnkle",11]
    ]
    for pair in correspondence:
      output_json["posenet"][pair[0]]["x"].append(pred_output_list[0][0]["pred_joints_smpl"][pair[1]][0])
      output_json["posenet"][pair[0]]["y"].append(pred_output_list[0][0]["pred_joints_smpl"][pair[1]][1])
      output_json["posenet"][pair[0]]["z"].append(pred_output_list[0][0]["pred_joints_smpl"][pair[1]][2])
    return output_json

def scale_joints(output_json,scale):
    size = len(data_dict['posenet']['nose']['x']
    for key in output_json.keys():
        for little_key in output_json[key].keys():
          for i in range(size):
            output_json[key][little_key]['x'][i] = (output_json[key][little_key]['x'][i] + 1.0)*scale[0]
            output_json[key][little_key]['y'][i] = (output_json[key][little_key]['y'][i] + 1.0)*scale[1]
            output_json[key][little_key]['z'][i] = (output_json[key][little_key]['z'][i] + 1.0)*scale[2]

def __filter_bbox_list(body_bbox_list, hand_bbox_list, single_person):
    # (to make the order as consistent as possible without tracking)
    bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
    idx_big2small = np.argsort(bbox_size)[::-1]
    body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
    hand_bbox_list = [hand_bbox_list[i] for i in idx_big2small]

    if single_person and len(body_bbox_list)>0:
        body_bbox_list = [body_bbox_list[0], ]
        hand_bbox_list = [hand_bbox_list[0], ]

    return body_bbox_list, hand_bbox_list


def run_regress(
    args, img_original_bgr, 
    body_bbox_list, hand_bbox_list, bbox_detector,
    body_mocap, hand_mocap,output_json
):
    cond1 = len(body_bbox_list) > 0 and len(hand_bbox_list) > 0
    cond2 = not args.frankmocap_fast_mode

    # use pre-computed bbox or use slow detection mode
    if cond1 or cond2:
        if not cond1 and cond2:
            # run detection only when bbox is not available
            body_pose_list, body_bbox_list, hand_bbox_list, _ = \
                bbox_detector.detect_hand_bbox(img_original_bgr.copy())
        else:
            print("Use pre-computed bounding boxes")
        assert len(body_bbox_list) == len(hand_bbox_list)

        if len(body_bbox_list) < 1: 
            return list(), list(), list()

        # sort the bbox using bbox size 
        # only keep on bbox if args.single_person is set
        body_bbox_list, hand_bbox_list = __filter_bbox_list(
            body_bbox_list, hand_bbox_list, args.single_person)

        # hand & body pose regression
        pred_hand_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True)
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(hand_bbox_list) == len(pred_hand_list)
        assert len(pred_hand_list) == len(pred_body_list)

    else:
        _, body_bbox_list = bbox_detector.detect_body_bbox(img_original_bgr.copy())

        if len(body_bbox_list) < 1: 
            return list(), list(), list()

        # sort the bbox using bbox size 
        # only keep on bbox if args.single_person is set
        hand_bbox_list = [None, ] * len(body_bbox_list)
        body_bbox_list, _ = __filter_bbox_list(
            body_bbox_list, hand_bbox_list, args.single_person)

        # body regression first 
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(body_bbox_list) == len(pred_body_list)

        # get hand bbox from body
        hand_bbox_list = body_mocap.get_hand_bboxes(pred_body_list, img_original_bgr.shape[:2])
        assert len(pred_body_list) == len(hand_bbox_list)

        # hand regression
        pred_hand_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True)
        assert len(hand_bbox_list) == len(pred_hand_list) 

    # intergration by copy-and-paste
    integral_output_list = intergration_copy_paste(
        pred_body_list, pred_hand_list, body_mocap.smpl, img_original_bgr.shape, output_json)
    
    return body_bbox_list, hand_bbox_list, integral_output_list


def run_frank_mocap(args, bbox_detector, body_mocap, hand_mocap, visualizer):
    #Setup input data to handle different types of inputs
    input_type, input_data = demo_utils.setup_input(args)
    cur_frame = args.start_frame
    video_frame = 0

    #Nossa estrutura de saida
    output_json = generate_json_structure()


    while True:
        # load data
        load_bbox = False

        if input_type =="image_dir":
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == "bbox_dir":
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]["image_path"]
                hand_bbox_list = input_data[cur_frame]["hand_bbox_list"]
                body_bbox_list = input_data[cur_frame]["body_bbox_list"]
                img_original_bgr  = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == "video":      
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue

            
          # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        
        elif input_type == "webcam":
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"
    
        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   
        print("--------------------------------------")
        
        # bbox detection
        if not load_bbox:
            body_bbox_list, hand_bbox_list = list(), list()
        
        # regression (includes integration)
        body_bbox_list, hand_bbox_list, pred_output_list = run_regress(
            args, img_original_bgr, 
            body_bbox_list, hand_bbox_list, bbox_detector,
            body_mocap, hand_mocap, output_json)
        #calculando escala da figura
        if(video_frame == 1):
          # height, width
            height = img_original_bgr.shape[0]
            width = img_original_bgr.shape[0]
            scalex = width/2
            scaley = height/2
            scalez = np.sqrt(scalex**2 + scaley**2)
            scale = [scalex,scaley,scalez]
        #Associando com nosso Json
        output_json = fill_body_joints(output_json,pred_output_list)
        #salvando nosso output em arquivo
    output_json = scale_joints(output_json,scale)
    json_name = str(args.input_path)[0:-4] + ".json"
    with open(json_name, "w") as outfile:
        output_json = str(output_json).replace("'",'"')
        json.dump(output_json,outfile)
def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert torch.cuda.is_available(), "Current version only supports GPU"

    hand_bbox_detector =  HandBboxDetector("third_view", device)
    
    #Set Mocap regressor
    body_mocap = BodyMocap(args.checkpoint_body_smplx, args.smpl_dir, device = device, use_smplx= True)
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)

    # Set Visualizer
    if args.renderer_type in ["pytorch3d", "opendr"]:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    run_frank_mocap(args, hand_bbox_detector, body_mocap, hand_mocap, visualizer)
  


if __name__ == "__main__":
    main()
