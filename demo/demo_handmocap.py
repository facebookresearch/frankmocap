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


def run_mocap_video(args, bbox_detector, hand_mocap):
    #Set up input data (images or webcam)
    image_list, cap = demo_utils.setup_input(args)
    sys.exit(0)

    cur_frame = args.startFrame -1
    while(True):
        # print("Start Mocap")
        # g_timer.tic()    

        cur_frame += 1        #starting from 0
        meshList =[]
        skelList =[]

        if len(image_list)>0:        #If the path is a folder
            if len(image_list)<=cur_frame:
                break
            elif args.endFrame>=0 and cur_frame > args.endFrame:
                break
            else:
                fName = image_list[cur_frame]
                img_original_bgr  = cv2.imread(fName)
        else:       #cap is None
            _, img_original_bgr = cap.read()
            fName = 'scene_{:08d}.pkl'.format(cur_frame)    

            if img_original_bgr is None: # Restart video at the end
                print("Warninig: img_original_bgr ==  None")
                # cap = cv2.VideoCapture(video_path)
                # ret, camInputFrame = cap.read()
                break   #Stop processing at the end of video

            if cap.isOpened()==False:
                print(">> Error: Input data is not valid or unavailable.")
                if args.url is not None:
                    print(">> Error: There would be version issues of your OpenCV in handling URL as the input stream")
                    print(">> Suggestion 1: Try to download the video via youtube-dl and put the video path as input")
                    print(">> Suggestion 2: Use --download or --d flag to automatically download and process it")
                    print("")
                assert False

        # Our operations on the frame come here
        # if cap is not None:  #If input from VideoCapture
        # img_original_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)          #Our model is trained with RGB
        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #Check existence of already processed data
        if args.skip and args.render_out_dir:
            assert False, "This code is not ready"
            mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName%cur_frame)
            if osp.exists(mergedImgFileName):
                print(f"Already exists: {mergedImgFileName}")
                continue


        ######################################################
        ## BBox detection

        bboxXYWH_list = bboxdetector.detectBbox(img_original_bgr)

        if args.single and len(bboxXYWH_list)>1:
            #Chose the biggest one
            diaSize =  [ (x[2]**2 + x[3]**2) for x in bboxXYWH_list]
            bigIdx = np.argmax(diaSize)
            bboxXYWH_list = [bboxXYWH_list[bigIdx]]

        g_debug_bboxonly= True
        if g_debug_bboxonly:
            # if False:#len(bboxXYWH_list)>0:
            #     for bbr in bboxXYWH_list:
            #         img_original_bgr = viewer2D.Vis_Bbox(img_original_bgr, bbr)
            #         viewer2D.ImShow(img_original_bgr, name="bboxDetect")
            # g_timer.toc(average =True, bPrint=True,title="DetectionTime")


            # Capture raw videos (to make a sample data)
            viewer2D.ImShow(img_original_bgr, name="bboxDetect")
            # mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName%cur_frame)
            # cv2.imwrite(mergedImgFileName, img_original_bgr)
            # continue
        # g_timer.toc(average =True, bPrint=True,title="Detect")
       
        ######################################################
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


def run_mocap_image(args, bbox_detector, hand_mocap):
    #Set up input data (images or webcam)
    image_list, _ = demo_utils.setup_input(args)

    for f_id, img_path in enumerate(image_list):

        if args.crop_type == 'hand_crop':
            img_original_bgr = cv2.imread(img_path)
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
            # g_utils.save_mesh_to_obj("origin.obj", pred_output['pred_vertices_origin'], pred_output['faces'])
            # g_utils.save_mesh_to_obj("vis.obj", pred_output['pred_vertices_img'], pred_output['faces'])
            # sys.exit(0)
        else:            
            pass

        sys.exit(0)

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


def main():
    args = DemoOptions().parse()
    print(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    bbox_detector =  HandBboxDetector('2dpose')      #"yolo" or "2dpose"

    SMPL_MODEL_DIR = '../data/smplx/'
    hand_mocap = HandMocap(args.checkpoint, SMPL_MODEL_DIR, device = device)

    if args.input_type == 'image':
        run_mocap_image(args, bbox_detector, hand_mocap)
    else:
        run_mocap_video(args, bbox_detector, hand_mocap)

if __name__ == '__main__':
    main()