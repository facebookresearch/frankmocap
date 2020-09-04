# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize

from .demo_options import DemoOptions
import mocap_utils.general_utils as g_utils
import mocap_utils.demo_utils as demo_utils

# from renderer import viewer2D #, glViewer
# from renderer.visualizer import Visualizer
from handmocap.mocap_api import HandMocap
from demo.demo_bbox_detector import HandBboxDetector
import renderer.opendr_renderer as od_render
from mocap_utils.vis_utils import Visualizer


def run_mocap_video(args, bbox_detector, hand_mocap):
    """
    Not implemented Yet.
    """
    pass


def run_mocap_image(args, bbox_detector, hand_mocap):
    #Set up input data (images or webcam)
    image_list, _ = demo_utils.setup_input(args)
    visualizer = Visualizer()

    for f_id, img_name in enumerate(image_list):
        img_path = osp.join(args.input_image_dir, img_name)

        # read images
        img_original_bgr = cv2.imread(img_path)

        if args.crop_type == 'hand_crop':
            pred_output = hand_mocap.regress(img_original_bgr, None, 'rhand')

            if args.renderer_type == "opendr":
                cam = np.zeros(3,)
                cam[0] = pred_output['cam_scale']
                cam[1:] = pred_output['cam_trans']
                bbox_scale_ratio = pred_output['bbox_scale_ratio']
                bbox_top_left = pred_output['bbox_top_left']
                verts = pred_output['pred_vertices_origin']
                faces = pred_output['faces']
                img = pred_output['img_cropped']

                rend_img0 = od_render.render(cam, verts, faces, bg_img=img)
                cv2.imwrite("0.png", rend_img0)
                rend_img1 = od_render.render_to_origin_img(cam, verts, faces, 
                    bg_img=img_original_bgr, bbox_scale=bbox_scale_ratio, bbox_top_left=bbox_top_left)
                cv2.imwrite("1.png", rend_img1)
                sys.exit(0)
            elif args.renderer_type == "opengl_no_gui":
                pass
            else:
                continue
        else:            
            # Input images has other body part or hand not cropped.
            assert args.crop_type == 'no_crop'
            assert args.view_type == 'third_view'
            body_pose_list, hand_bbox_list, raw_hand_bboxes = bbox_detector.detect_hand_bbox(img_original_bgr.copy())

            vis_img = visualizer.visualize(
                input_img = img_original_bgr.copy(), 
                hand_bbox_list = hand_bbox_list,
                body_pose_list = body_pose_list,
                raw_hand_bboxes = raw_hand_bboxes
            )
            res_img_path = osp.join(args.render_out_dir, img_name)
            g_utils.make_subdir(res_img_path)
            cv2.imwrite(res_img_path, vis_img)
            print(f"Image path: {img_path}")
            # body_pose = bbox_detector.model.detect_body_pose(img_original_bgr.copy())
            # left_arm_img = od_render.draw_keypoints(img_original_bgr, body_pose[0][5:8, :], color=(255,0,0), radius=10)
            # right_arm_img = od_render.draw_keypoints(left_arm_img, body_pose[0][2:5, :], color=(0,0,255), radius=10)

    '''
        continue
        ## Body Pose Regression
        if len(bboxXYWH_list)>0:

            pred_rotmat_all =[]
            pred_betas_all =[]
            pred_camera_all =[]
            pred_vertices_all =[]
            pred_joints_3d_all =[]
            bbox_all =[]
            boxScale_o2n_all =[]
            bboxTopLeft_all =[]

            for bboxXYHW in bboxXYWH_list:
                predoutput ={}
                for lr  in bboxXYHW:        #'lhand' or 'rhand'

                    if bboxXYHW[lr] is None:    #if Not None
                        predoutput[lr] = None
                        continue
                    
                    predoutput[lr] = hand_mocap.regress(img_original_bgr, bboxXYHW[lr], lr)

                    if predoutput is None:
                        continue
                    pred_vertices_img = predoutput[lr]['pred_vertices_img']
                    # pred_joints_img = predoutput['pred_joints_img']
                    
                    if lr =='lhand':
                        tempMesh = {'ver': pred_vertices_img, 'f':  hand_mocap.lhand_mesh_face}
                    else:
                        tempMesh = {'ver': pred_vertices_img, 'f':  hand_mocap.rhand_mesh_face}
                    meshList.append(tempMesh)
                    # skelList.append(pred_joints_img.ravel()[:,np.newaxis])  #(49x3, 1)

                if args.pklout:
                    pred_rotmat_all.append(predoutput['pred_rotmat'])
                    pred_betas_all.append(predoutput['pred_betas'])
                    pred_camera_all.append(predoutput['pred_camera'])
                    pred_vertices_all.append(pred_vertices_img)
                    pred_joints_3d_all.append(pred_joints_img)
                    bbox_all.append(predoutput['bbox_xyxy'])

                    bboxTopLeft_all.append(predoutput['bboxTopLeft'])
                    boxScale_o2n_all.append(predoutput['boxScale_o2n'])
        
            ######################################################
            ## Export to pkl
            if args.pklout and len(pred_rotmat_all)>0:
                pred_rotmat_all = np.concatenate(pred_rotmat_all,axis=0)
                pred_betas_all = np.concatenate(pred_betas_all,axis=0)
                pred_camera_all = np.concatenate(pred_camera_all)
                pred_vertices_all = np.concatenate(pred_vertices_all)
                pred_joints_3d_all = np.concatenate(pred_joints_3d_all)
                # bbox_all = np.concatenate(bbox_all)
                # bboxTopLeft_all = np.concatenate(bboxTopLeft_all)
                # boxScale_o2n_all =np.concatenate(boxScale_o2n_all)
                dataOut = {
                    'pred_rotmat_all': pred_rotmat_all,
                    'pred_betas_all': pred_betas_all,
                    'cams_person': pred_camera_all,
                    'pred_joints_3d_all': pred_joints_3d_all,
                    'verts_person_og':pred_vertices_all,
                    'boxScale_o2n_all': boxScale_o2n_all,
                    'bboxTopLeft_all': bboxTopLeft_all,
                    'bbox':bbox_all
                }

                if args.render_out_dir is None:
                    print("Please set output folder by --out")
                    assert False
                    
                else:
                    mocapOutFolder = osp.join(args.render_out_dir, 'mocap')
                    g_utils.build_dir(mocapOutFolder)
                    outputFileName_pkl = osp.join(mocapOutFolder,osp.basename(fName)[:-4]+'.pkl')
                    fout = open(outputFileName_pkl,'wb')
                    pickle.dump(dataOut, fout)
                    fout.close()
        
        # g_timer.toc(average =True, bPrint=True,title="Detect+Regress")
        ######################################################
        ## Visualization

        if args.noVis == False:        #Visualize
            # img_original  = img_original_bgr[:,:,[2,1,0]]
            # img_original = np.ascontiguousarray(img_original, dtype=np.uint8)
            assert img_original_bgr.shape[0]>0 and img_original_bgr.shape[1]>0

            #Render output to files            
            if renderOutRoot:
                visualizer.visualize_screenless_naive(meshList, skelList, bboxXYWH_list, img_original_bgr)

                overlaidImg = visualizer.renderout['render_camview']
                overlaidImgFileName = '{0}/{1}'.format(overlaidImageFolder,outputFileName%cur_frame)
                cv2.imwrite(overlaidImgFileName, overlaidImg)

                sideImg = visualizer.renderout['render_sideview']
                sideImgFileName = '{0}/{1}'.format(sideImageFolder,outputFileName%cur_frame)
                cv2.imwrite(sideImgFileName, sideImg)

                if True:    #merged view rendering
                    # overlaidImg_resized = cv2.resize(overlaidImg, (img_original_bgr.shape[1], img_original_bgr.shape[0]))
                    img_original_bgr_resized = cv2.resize(img_original_bgr, (overlaidImg.shape[1], overlaidImg.shape[0]))
                    sideImg_resized = cv2.resize(sideImg, (overlaidImg.shape[1], overlaidImg.shape[0]))
                    mergedImg = np.concatenate( (img_original_bgr_resized, overlaidImg, sideImg_resized), axis=1)
                    viewer2D.ImShow(mergedImg,name="merged")

                    # viewer2D.ImShow(overlaidImg)
                    mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName%cur_frame)
                    cv2.imwrite(mergedImgFileName, mergedImg)
                    print(f"Saved to {mergedImgFileName}")

            #Do not save files but jut GUI visualization
            else:
                visualizer.visualize_gui_naive(meshList, skelList, [], img_original_bgr)
        # g_timer.toc(average =True, bPrint=True,title="Detect+Regress+Vis")

    # When everything done, release the capture
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    # Video generation from rendered images
    if args.noVis == False and args.noVideoOut==False:
        if renderOutRoot and osp.exists( osp.join(renderOutRoot, 'merged') ): 
            print(">> Generating video in {}/{}.mp4".format(renderOutRoot,osp.basename(renderOutRoot) ))
            inputFrameDir = osp.join(renderOutRoot, 'merged')
            outVideo_fileName = osp.join(renderOutRoot, osp.basename(renderOutRoot)+'.mp4')
            ffmpeg_cmd = 'ffmpeg -y -f image2 -framerate 25 -pattern_type glob -i "{0}/*.jpg"  -pix_fmt yuv420p -c:v libx264 -x264opts keyint=25:min-keyint=25:scenecut=-1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {1}'.format(inputFrameDir, outVideo_fileName)
            os.system(ffmpeg_cmd)
    '''


def main():
    args = DemoOptions().parse()
    print(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    bbox_detector =  HandBboxDetector(args.view_type, device)

    SMPL_MODEL_DIR = './data/smplx/'
    hand_mocap = HandMocap(args.checkpoint, SMPL_MODEL_DIR, device = device)

    if args.input_type == 'image':
        run_mocap_image(args, bbox_detector, hand_mocap)
    else:
        run_mocap_video(args, bbox_detector, hand_mocap)

if __name__ == '__main__':
    main()