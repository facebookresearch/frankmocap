# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import pickle

############# input parameters  #############
default_checkpoint ='models_eft/2020_05_31-00_50_43-best-51.749683916568756.pt'

from renderer import viewer2D#, glViewer
from renderer.visualizer import Visualizer

from handmocap.mocap_api import HandMocap
from demo.demo_bbox_detector import HandBboxDetector
# from bodymocap.utils.timer import Timer
# g_timer = Timer()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=False, default=default_checkpoint, help='Path to pretrained checkpoint')
parser.add_argument('--vPath', type=str, default=None, help="""Path of video or first image in a folder
                    (example: (path)/out%%1d.jpg - %%1d will be automatically replaced by the number of the image)
                    . Can also be used to load a single image (example: (path)/out1.jpg).""")
parser.add_argument('--webcam', '-W', action='store_true', help='Use webcam for video.')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--renderout', type=str, default=None, help='Folder of output images.')
parser.add_argument('--pklout', action='store_true', help='Export mocap output as pkl file')
parser.add_argument('--url', '-U', type=str, default=None, help='URL of YouTube video, or image.')
parser.add_argument('--bUseSMPLX', action='store_true', help='use SMPLX instead of SMPL. You should use a model trained with SMPL-X')
parser.add_argument('--download', '-d', action='store_true', help='Download YouTube video first (in webvideo folder), and process it')
parser.add_argument('--noVis', action='store_true', help='Do not visualize output on the screen')
parser.add_argument('--startFrame', type=int, default=0, help='given a sequence of frames, set the starting frame')
parser.add_argument('--endFrame', type=int, default=-1, help='given a sequence of frames, set the last frame')
parser.add_argument('--noVideoOut', action='store_true', help='Do not generate output video (ffmpeg)')
parser.add_argument('--single', action='store_true', help='Reconstruct only one person in the scene with the biggest bbox')
parser.add_argument('--skip', action='store_true', help='Skip there exist already processed outputs')

def get_video_path(args):
    if args.webcam:
        video_path = 0
    elif args.url:
        if args.download:
            os.makedirs("./webvideos",exist_ok=True)
            downloadPath ="./webvideos/{0}.mp4".format(os.path.basename(args.url))
            cmd_download = "youtube-dl -f best {0} -o {1}".format(args.url,downloadPath)
            print(">> Downloading: {}".format(args.url))
            print(">> {}".format(cmd_download))
            #download via youtube-dl
            os.system(cmd_download)
            video_path = downloadPath
        else:
            try:
                import pafy
                url = args.url #'https://www.youtube.com/watch?v=c5nhWy7Zoxg'
                vPafy = pafy.new(url)
                play = vPafy.getbest(preftype="webm")
                video_path = play.url
                video_path = url
            except:
                video_path = args.url
    elif args.vPath:
        video_path = args.vPath
    else:
        assert False
    return video_path


def RunMonomocap(args, video_path, visualizer, bboxdetector, handmocap, device, renderOutRoot):

    #Set up output folders
    if renderOutRoot:
        outputFileName = 'scene_%08d.jpg' # Hardcoded in glViewer.py
        if os.path.exists(renderOutRoot)==False:
            os.mkdir(renderOutRoot)

        overlaidImageFolder= os.path.join(renderOutRoot, 'overlaid')
        if os.path.exists(overlaidImageFolder)==False:
            os.mkdir(overlaidImageFolder)

        sideImageFolder= os.path.join(renderOutRoot, 'side')
        if os.path.exists(sideImageFolder)==False:
            os.mkdir(sideImageFolder)

        mergedImageFolder= os.path.join(renderOutRoot, 'merged')
        if os.path.exists(mergedImageFolder)==False:
            os.mkdir(mergedImageFolder)

        g_renderDir= os.path.join(renderOutRoot, 'render')
        if os.path.exists(g_renderDir)==False:
            os.mkdir(g_renderDir)

    #Set up input data (images or webcam)
    image_list =[]
    cap =None
    if os.path.isdir(video_path):       #if video_path is a dir, load all videos
        image_list = sorted(os.listdir(video_path))
        image_list = [os.path.join(video_path,f) for f in image_list]
    else:
        cap = cv2.VideoCapture(video_path)
        if os.path.exists(video_path):
            print("valid")
        if cap.isOpened()==False:
            print(f"Failed in opening video: {video_path}")
            assert False

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
        if args.skip and renderOutRoot:
            # viewer2D.ImShow(overlaidImg)
            mergedImgFileName = '{0}/{1}'.format(mergedImageFolder,outputFileName%cur_frame)
            if os.path.exists(mergedImgFileName):
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
                    
                    predoutput[lr] = handmocap.regress(img_original_bgr, bboxXYHW[lr], lr)

                    if predoutput is None:
                        continue
                    pred_vertices_img = predoutput[lr]['pred_vertices_img']
                    # pred_joints_img = predoutput['pred_joints_img']
                    
                    if lr =='lhand':
                        tempMesh = {'ver': pred_vertices_img, 'f':  handmocap.lhand_mesh_face}
                    else:
                        tempMesh = {'ver': pred_vertices_img, 'f':  handmocap.rhand_mesh_face}
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

                if renderOutRoot is None:
                    print("Please set output folder by --out")
                    assert False
                    
                else:
                    mocapOutFolder = os.path.join(renderOutRoot,'mocap')
                    if not os.path.exists(mocapOutFolder):
                        os.mkdir(mocapOutFolder)

                    outputFileName_pkl = os.path.join(mocapOutFolder,os.path.basename(fName)[:-4]+'.pkl')
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
        if renderOutRoot and os.path.exists( os.path.join(renderOutRoot, 'merged') ): 
            print(">> Generating video in {}/{}.mp4".format(renderOutRoot,os.path.basename(renderOutRoot) ))
            inputFrameDir = os.path.join(renderOutRoot, 'merged')
            outVideo_fileName = os.path.join(renderOutRoot, os.path.basename(renderOutRoot)+'.mp4')
            ffmpeg_cmd = 'ffmpeg -y -f image2 -framerate 25 -pattern_type glob -i "{0}/*.jpg"  -pix_fmt yuv420p -c:v libx264 -x264opts keyint=25:min-keyint=25:scenecut=-1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {1}'.format(inputFrameDir, outVideo_fileName)
            os.system(ffmpeg_cmd)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    checkpoint = args.checkpoint
    video_path = get_video_path(args)
    renderOutRoot = args.renderout

    if renderOutRoot:
        visualizer = Visualizer('nongui')
    else:
        visualizer = Visualizer('gui')
    bboxdetector =  HandBboxDetector('2dpose')      #"yolo" or "2dpose"

    SMPL_MODEL_DIR = 'smplx_path'
    handymocap = HandMocap(args.checkpoint, SMPL_MODEL_DIR, device = device)

    RunMonomocap(args, video_path, visualizer, bboxdetector, handymocap, device, renderOutRoot)