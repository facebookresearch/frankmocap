# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import cv2
from collections import OrderedDict
import mocap_utils.general_utils as gnu
import numpy as np
import json
import subprocess as sp


def setup_render_out(out_dir):
    if out_dir is not None:
        gnu.build_dir(out_dir)
        outputFileName = 'scene_%08d.jpg' # Hardcoded in glViewer.py

        overlaidImageFolder= osp.join(out_dir, 'overlaid')
        gnu.build_dir(overlaidImageFolder)

        sideImageFolder= osp.join(out_dir, 'side')
        gnu.build_dir(sideImageFolder)

        mergedImageFolder= osp.join(out_dir, 'merged')
        gnu.build_dir(mergedImageFolder)

        res_subdirs = \
            [outputFileName, overlaidImageFolder, sideImageFolder, mergedImageFolder]
        return res_subdirs
    
    else:
        return None


def __get_input_type(args):
    input_type =None
    image_exts = ('jpg', 'png', 'jpeg', 'bmp')
    video_exts = ('mp4', 'avi', 'mov')
    extension = osp.splitext(args.input_path)[1][1:]

    if extension.lower() in video_exts:
        input_type ='video'
    elif osp.isdir(args.input_path):
        file_list = os.listdir(args.input_path)
        assert len(file_list) >0, f"{args.input_path} is a blank folder"
        extension = osp.splitext(file_list[0])[1][1:]
        if extension == 'json':
            input_type ='bbox_dir'
        else:
            assert extension.lower() in image_exts
            input_type ='image_dir'
    elif args.input_path =='webcam':
        input_type ='webcam'
    else:
        assert False, "Unknown input path. It should be an image," + \
            "or an image folder, or a video file, or \'webcam\' "
    return input_type


def __video_setup(args):
    video_path = args.input_path
    video_dir, video_name, video_basename, ext = gnu.analyze_path(video_path)
    args.seq_name = video_basename

    if args.save_frame:
        frame_dir = osp.join(args.out_dir, "frames")
        gnu.build_dir(frame_dir)    

    render_out_dir = osp.join(args.out_dir, "rendered")
    gnu.build_dir(render_out_dir)

    mocap_out_dir = osp.join(args.out_dir, "mocap")
    gnu.build_dir(mocap_out_dir)


def __img_seq_setup(args):
    seq_dir_path = args.input_path
    args.seq_name = os.path.basename(args.input_path)

    render_out_dir = osp.join(args.out_dir, 'rendered')
    gnu.build_dir(render_out_dir)

    mocap_out_dir = osp.join(args.out_dir, "mocap")
    gnu.build_dir(mocap_out_dir)


def setup_input(args):
    """
    Input type can be 
        an image file
        a video file
        a folder with image files
        a folder with bbox (json) files
        "webcam"
    
    """
    image_exts = ('jpg', 'png', 'jpeg', 'bmp')
    video_exts = ('mp4', 'avi', 'mov')

    # get type of input 
    input_type = __get_input_type(args)

    if input_type =='video':
        cap = cv2.VideoCapture(args.input_path)
        assert cap.isOpened(), f"Failed in opening video: {args.input_path}"
        __video_setup(args)
        return input_type, cap

    elif input_type =='webcam':
        cap = cv2.VideoCapture(0)       #webcam input
        return input_type, cap

    elif input_type =='image_dir':
        image_list = gnu.get_all_files(args.input_path, image_exts, "relative") 
        image_list = [ osp.join(args.input_path, image_name) for image_name in image_list ]
        __img_seq_setup(args)
        return input_type, image_list

    elif input_type =='bbox_dir':
        __img_seq_setup(args)
        json_files = gnu.get_all_files(args.input_path, '.json', "relative") 
        input_data = list()
        for json_file in json_files:
            json_path = osp.join(args.input_path, json_file)
            image_path, body_bbox_list, hand_bbox_list = load_info_from_json(json_path)
            input_data.append(dict(
                image_path = image_path,
                hand_bbox_list = hand_bbox_list,
                body_bbox_list = body_bbox_list
            ))
        return input_type, input_data

    else:
        assert False, "Unknown input type"


def extract_mesh_from_output(pred_output_list):
    pred_mesh_list = list()
    for pred_output in pred_output_list:
        if pred_output is not None:
            if 'left_hand' in pred_output: # hand mocap
                for hand_type in pred_output:
                    if pred_output[hand_type] is not None:
                        vertices = pred_output[hand_type]['pred_vertices_img']
                        faces = pred_output[hand_type]['faces'].astype(np.int32)
                        pred_mesh_list.append(dict(
                            vertices = vertices,
                            faces = faces
                        ))
            else: # body mocap (includes frank/whole/total mocap)
                vertices = pred_output['pred_vertices_img']
                faces = pred_output['faces'].astype(np.int32)
                pred_mesh_list.append(dict(
                    vertices = vertices,
                    faces = faces
                ))
    return pred_mesh_list
                
    
def load_info_from_json(json_path):
    data = gnu.load_json(json_path)
    # image path
    assert ('image_path' in data), "Path of input image should be specified"
    image_path = data['image_path']
    assert osp.exists(image_path), f"{image_path} does not exists"
    # body bboxes
    body_bbox_list = list()
    if 'body_bbox_list' in data:
        body_bbox_list = data['body_bbox_list']
        assert isinstance(body_bbox_list, list)
        for b_id, body_bbox in enumerate(body_bbox_list):
            if isinstance(body_bbox, list) and len(body_bbox) == 4:
                body_bbox_list[b_id] = np.array(body_bbox)
    # hand bboxes
    hand_bbox_list = list()
    if 'hand_bbox_list' in data:
        hand_bbox_list = data['hand_bbox_list']
        assert isinstance(hand_bbox_list, list)
        for hand_bbox in hand_bbox_list:
            for hand_type in ['left_hand', 'right_hand']:
                if hand_type in hand_bbox:
                    bbox = hand_bbox[hand_type]
                    if isinstance(bbox, list) and len(bbox) == 4:
                        hand_bbox[hand_type] = np.array(bbox)
                    else:
                        hand_bbox[hand_type] = None
    return image_path, body_bbox_list, hand_bbox_list


def save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list):
    saved_data = dict()

    # image_path
    saved_data['image_path'] = image_path 

    # body_bbox_list
    saved_body_bbox_list = list()
    for body_bbox in body_bbox_list:
        if body_bbox is not None:
            saved_body_bbox_list.append(body_bbox.tolist())
    saved_data['body_bbox_list'] = saved_body_bbox_list

    # hand_bbox_list
    saved_hand_bbox_list = list()
    for hand_bbox in hand_bbox_list:
        if hand_bbox is not None:
            saved_hand_bbox = dict(
                left_hand = None,
                right_hand = None)
            for hand_type in saved_hand_bbox:
                bbox = hand_bbox[hand_type]
                if bbox is not None:
                    saved_hand_bbox[hand_type] = bbox.tolist()
            saved_hand_bbox_list.append(saved_hand_bbox)
    saved_data['hand_bbox_list'] = saved_hand_bbox_list

    # write data to json
    img_name = osp.basename(image_path)
    record = img_name.split('.')
    json_name = f"{'.'.join(record[:-1])}_bbox.json"
    json_path = osp.join(args.out_dir, 'bbox', json_name)
    gnu.make_subdir(json_path)
    gnu.save_json(json_path, saved_data)
    print(f"Bbox saved: {json_path}")


def save_pred_to_pkl(
    args, demo_type, image_path, 
    body_bbox_list, hand_bbox_list, pred_output_list):

    smpl_type = 'smplx' if args.use_smplx else 'smpl'
    assert demo_type in ['hand', 'body', 'frank']
    if demo_type in ['hand', 'frank']:
        assert smpl_type == 'smplx'

    assert len(hand_bbox_list) == len(body_bbox_list)
    assert len(body_bbox_list) == len(pred_output_list)

    saved_data = dict()
    # demo type / smpl type / image / bbox
    saved_data = OrderedDict()
    saved_data['demo_type'] = demo_type
    saved_data['smpl_type'] = smpl_type
    saved_data['image_path'] = osp.abspath(image_path)
    saved_data['body_bbox_list'] = body_bbox_list
    saved_data['hand_bbox_list'] = hand_bbox_list
    saved_data['save_mesh'] = args.save_mesh

    saved_data['pred_output_list'] = list()
    num_subject = len(hand_bbox_list)
    for s_id in range(num_subject):
        # predict params
        pred_output = pred_output_list[s_id]
        if pred_output is None:
            saved_pred_output = None
        else:
            saved_pred_output = dict()
            if demo_type == 'hand':
                for hand_type in ['left_hand', 'right_hand']:
                    pred_hand = pred_output[hand_type]
                    saved_pred_output[hand_type] = dict()
                    saved_data_hand = saved_pred_output[hand_type]
                    if pred_hand is None:
                        saved_data_hand = None
                    else:
                        for pred_key in pred_hand:
                            if pred_key.find("vertices")<0 or pred_key == 'faces' :
                                saved_data_hand[pred_key] = pred_hand[pred_key]
                            else:
                                if args.save_mesh:
                                    if pred_key != 'faces':
                                        saved_data_hand[pred_key] = \
                                            pred_hand[pred_key].astype(np.float16)
                                    else:
                                        saved_data_hand[pred_key] = pred_hand[pred_key]
            else:
                for pred_key in pred_output:
                    if pred_key.find("vertices")<0 or pred_key == 'faces' :
                        saved_pred_output[pred_key] = pred_output[pred_key]
                    else:
                        if args.save_mesh:
                            if pred_key != 'faces':
                                saved_pred_output[pred_key] = \
                                    pred_output[pred_key].astype(np.float16)
                            else:
                                saved_pred_output[pred_key] = pred_output[pred_key]

        saved_data['pred_output_list'].append(saved_pred_output)

    # write data to pkl
    img_name = osp.basename(image_path)
    record = img_name.split('.')
    pkl_name = f"{'.'.join(record[:-1])}_prediction_result.pkl"
    pkl_path = osp.join(args.out_dir, 'mocap', pkl_name)
    gnu.make_subdir(pkl_path)
    gnu.save_pkl(pkl_path, saved_data)
    print(f"Prediction saved: {pkl_path}")
 

def save_res_img(out_dir, image_path, res_img):
    out_dir = osp.join(out_dir, "rendered")
    img_name = osp.basename(image_path)
    img_name = img_name[:-4] + '.jpg'           #Always save as jpg
    res_img_path = osp.join(out_dir, img_name)
    gnu.make_subdir(res_img_path)
    cv2.imwrite(res_img_path, res_img)
    print(f"Visualization saved: {res_img_path}")


def gen_video_out(out_dir, seq_name):
    outVideo_fileName = osp.join(out_dir, seq_name+'.mp4')
    print(f">> Generating video in {outVideo_fileName}")

    in_dir = osp.abspath(osp.join(out_dir, "rendered"))
    out_path = osp.abspath(osp.join(out_dir, seq_name+'.mp4'))
    ffmpeg_cmd = f'ffmpeg -y -f image2 -framerate 25 -pattern_type glob -i "{in_dir}/*.jpg"  -pix_fmt yuv420p -c:v libx264 -x264opts keyint=25:min-keyint=25:scenecut=-1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {out_path}'
    os.system(ffmpeg_cmd)
    # print(ffmpeg_cmd.split())
    # sp.run(ffmpeg_cmd.split())
    # sp.Popen(ffmpeg_cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
