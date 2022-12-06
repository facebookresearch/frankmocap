import argparse
import os.path

import cv2
import numpy as np

import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer

from renderer.viewer2D import ImShow


def input_frame_and_metadata_iterator(args):
    # Setup input data to handle different types of inputs
    input_type, input_data = demo_utils.setup_input(args)

    assert args.out_dir is not None, "Please specify output dir to store the results"
    cur_frame = args.start_frame
    video_frame = 0
    timer = Timer()

    while True:
        timer.tic()

        # load data
        load_bbox = False
        hand_bbox_list = None
        body_bbox_list = None
        image_path = None

        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                print("Use pre-computed bounding boxes")
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = os.path.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)

        elif input_type == 'webcam':
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = os.path.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame += 1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break

        input_frame_and_metadata = argparse.Namespace(
            image_path=image_path,
            img_original_bgr=img_original_bgr,
            load_bbox=load_bbox,
        )

        print("--------------------------------------")

        if load_bbox:
            input_data.body_bbox_list = body_bbox_list
            input_data.hand_bbox_list = hand_bbox_list
        yield input_frame_and_metadata

        timer.toc(bPrint=True, title="Time")
        print(f"Processed : {image_path}")

    # save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    # When everything done, release the capture
    if input_type == 'webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()


def detect_hand_bbox_and_save_it_into_frame_and_metadata(args, input_frame_and_metadata, bbox_detector_method):
    image_path = input_frame_and_metadata.image_path
    img_original_bgr = input_frame_and_metadata.img_original_bgr
    load_bbox = input_frame_and_metadata.load_bbox

    # bbox detection
    body_bbox_list = None
    if load_bbox:
        body_bbox_list = input_frame_and_metadata.body_bbox_list
        hand_bbox_list = input_frame_and_metadata.hand_bbox_list
        body_pose_list = None
        raw_hand_bboxes = None
    elif args.crop_type == 'hand_crop':
        # hand already cropped, therefore, no need for detection
        img_h, img_w = img_original_bgr.shape[:2]
        body_pose_list = None
        raw_hand_bboxes = None
        hand_bbox_list = [dict(right_hand=np.array([0, 0, img_w, img_h]))]
    else:
        # Input images has other body part or hand not cropped.
        # Use hand detection model & body detector for hand detection
        assert args.crop_type == 'no_crop'
        detect_output = bbox_detector_method(img_original_bgr.copy())
        body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output

    # save the obtained body & hand bbox to json file
    if args.save_bbox_output:
        demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

    if len(hand_bbox_list) < 1:
        print(f"No hand detected: {image_path}")
        return False

    input_frame_and_metadata.body_bbox_list = body_bbox_list
    input_frame_and_metadata.hand_bbox_list = hand_bbox_list
    input_frame_and_metadata.body_pose_list = body_pose_list
    input_frame_and_metadata.raw_hand_bboxes = raw_hand_bboxes
    return True


def show_and_save_result(
        args, demo_type, input_frame_and_metadata, visualizer=None,
        pred_output_list=None, transformed_image=None, image_category="rendered"
):
    image_path = input_frame_and_metadata.image_path
    img_original_bgr = input_frame_and_metadata.img_original_bgr
    body_bbox_list = input_frame_and_metadata.body_bbox_list
    hand_bbox_list = input_frame_and_metadata.hand_bbox_list

    # extract mesh for rendering (vertices in image space and faces) from pred_output_list
    pred_mesh_list = None
    if pred_output_list is not None:
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

    res_img = None
    if transformed_image is not None:
        res_img = transformed_image
    elif visualizer is not None:
        # visualization
        if demo_type == 'frank':
            res_img = visualizer.visualize(
                img_original_bgr,
                pred_mesh_list=pred_mesh_list,
                body_bbox_list=body_bbox_list,
                hand_bbox_list=hand_bbox_list
            )
        elif demo_type == 'body':
            res_img = visualizer.visualize(
                img_original_bgr,
                pred_mesh_list=pred_mesh_list,
                hand_bbox_list=body_bbox_list
            )
        elif demo_type == 'hand':
            res_img = visualizer.visualize(
                img_original_bgr,
                pred_mesh_list=pred_mesh_list,
                hand_bbox_list=hand_bbox_list
            )
        else:
            raise ValueError("Unknown demo_type")

    if res_img is not None:
        # show result in the screen
        if not args.no_display:
            res_img = res_img.astype(np.uint8)
            ImShow(res_img)

        # save result image (we can make an option here)
        if args.out_dir is not None:
            demo_utils.save_res_img(
                args.out_dir, image_path, res_img, image_category=image_category
            )

    # save predictions to pkl
    if args.save_pred_pkl:
        demo_utils.save_pred_to_pkl(
            args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list
        )
