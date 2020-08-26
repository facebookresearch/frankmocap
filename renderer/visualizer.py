# Copyright (c) Facebook, Inc. and its affiliates.

"""
Visualizing 3D humans via Opengl
- Options:
    GUI mode: a screen is required
    Scnreenless mode: xvfb-run can be used to avoid screen requirement

"""
import numpy as np
import cv2
import torch
from renderer import viewer2D#, glViewer, glRenderer
from renderer import meshRenderer #glRenderer
from renderer import glViewer #glRenderer

# from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm


class Visualizer(object):
    """
   Visualizer to visuzlie SMPL reconstruction output from HMR family (HMR, SPIN, EFT)

    Args:
        reconstruction output
        rawImg, bbox, 
        smpl_params (shape, pose, cams )
    """

    def __init__(
        self,
        rendererType ='gui'          #nongui or gui
    ):
        self.rendererType = rendererType
        if rendererType != "gui" and rendererType!= "nongui":
            print("Wrong rendererType: {rendererType}")
            assert False

        self.cam_all = []
        self.vert_all = []
        self.bboxXYXY_all = []

        self.bg_image = None

        #Screenless rendering
        if rendererType =='nongui':
            self.renderer = meshRenderer.meshRenderer()
            self.renderer.setRenderMode('geo')
            self.renderer.offscreenMode(True)
        else:
            self.renderer = None

        #Output rendering
        self.renderout = None

    # def setSMPLParam(self, smpl_vertices, cam, bbox_xyxy):
    #     """
    #         smpl_vertices: (6890,3)
    #         cam: (3,)
    #         bbox_xyxy: (3,)
    #     """

    #     self.cam_all.append(smpl_vertices)
    #     self.vert_all.append(cam)
    #     self.bboxXYXY_all.append(bbox_xyxy)

    # def setImg(self, image):
    #     self.bg_image = image

    # def setWindowSize(width_, height_):

    #     if self.rendererType=="gui":
    #         glViewer.setWindowSize(width_, height_)
    #     else:
    #         assert False


    # def visualize(self, image, instances, mesh_fname=""):
    #     """
    #     """
    #     pred_camera  = self.cam_all[0]
    #     camParam_scale = pred_camera[0]
    #     camParam_trans = pred_camera[1:]

    #     pred_vert_vis = self.vert_all[0]

    #     # smpl_joints_3d_vis = smpl_joints_3d

    #     draw_onbbox = True
    #     draw_rawimg = True

    #     if draw_onbbox:
    #         pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
    #         smpl_joints_3d_vis = convert_smpl_to_bbox(smpl_joints_3d_vis, camParam_scale, camParam_trans)
    #         renderer.setBackgroundTexture(croppedImg)
    #         renderer.setViewportSize(croppedImg.shape[1], croppedImg.shape[0])

    #         pred_vert_vis *=MAGNIFY_RATIO
        
    #     if draw_rawimg:
    #         #Covert SMPL to BBox first
    #         pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
    #         smpl_joints_3d_vis = convert_smpl_to_bbox(smpl_joints_3d_vis, camParam_scale, camParam_trans)

    #         #From cropped space to original
    #         pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0]) 
    #         smpl_joints_3d_vis = convert_bbox_to_oriIm(smpl_joints_3d_vis, boxScale_o2n, bboxTopLeft, rawImg.shape[1], rawImg.shape[0])
    #         renderer.setBackgroundTexture(rawImg)
    #         renderer.setViewportSize(rawImg.shape[1], rawImg.shape[0])

    #         #In orthographic model. XY of 3D is just 2D projection
    #         smpl_joints_2d_vis = conv_3djoint_2djoint(smpl_joints_3d_vis,rawImg.shape )
    #         # image_2dkeypoint_pred = viewer2D.Vis_Skeleton_2D_smpl45(smpl_joints_2d_vis, image=rawImg.copy(),color=(0,255,255))
    #         image_2dkeypoint_pred = viewer2D.Vis_Skeleton_2D_Openpose18(smpl_joints_2d_vis, image=rawImg.copy(),color=(255,0,0))        #All 2D joint
    #         image_2dkeypoint_pred = viewer2D.Vis_Skeleton_2D_Openpose18(smpl_joints_2d_vis, pt2d_visibility=keypoint_2d_validity, image=image_2dkeypoint_pred,color=(0,255,255))        #Only valid
    #         viewer2D.ImShow(image_2dkeypoint_pred, name='keypoint_2d_pred', waitTime=1)

    #     pred_meshes = {'ver': pred_vert_vis, 'f': smpl.faces}
    #     v = pred_meshes['ver'] 
    #     f = pred_meshes['f']

    #     #Visualize in the original image space
    #     renderer.set_mesh(v,f)
    #     renderer.showBackground(True)
    #     renderer.setWorldCenterBySceneCenter()
    #     renderer.setCameraViewMode("cam")

    #     #Set image size for rendering
    #     if args.onbbox:
    #         renderer.setViewportSize(croppedImg.shape[1], croppedImg.shape[0])
    #     else:
    #         renderer.setViewportSize(rawImg.shape[1], rawImg.shape[0])
            
    #     renderer.display()
    #     renderImg = renderer.get_screen_color_ibgr()
    #     viewer2D.ImShow(renderImg,waitTime=1)
        
    
    def visualize_screenless_naive(self, meshList, skelList, bboxXYWH_list, img_original, vis=False, maxHeight = 1080):
        
        """
            args:
                meshList: list of {'ver': pred_vertices, 'f': smpl.faces}
                skelList: list of [JointNum*3, 1]       (where 1 means num. of frames in glviewer)
                bbr_list: list of [x,y,w,h] 
            output:
                #Rendered images are saved in 
                self.renderout['render_camview']
                self.renderout['render_sideview']

            #Note: The size of opengl rendering is restricted by the current screen size. Set the maxHeight accordingly

        """
        assert self.renderer is not None

        if len(meshList)==0:
               # sideImg = cv2.resize(sideImg, (renderImg.shape[1], renderImg.shape[0]) )
            self.renderout  ={}
            self.renderout['render_camview'] = img_original.copy()

            blank = np.ones(img_original.shape, dtype=np.uint8)*255       #generate blank image
            self.renderout['render_sideview'] = blank
            return
        
        if len(bboxXYWH_list)>0:
            for bbr in bboxXYWH_list:
                viewer2D.Vis_Bbox(img_original,bbr)
        viewer2D.ImShow(img_original)

        #Check image height
        imgHeight, imgWidth = img_original.shape[0], img_original.shape[1]
        if maxHeight <imgHeight:        #Resize
            ratio = maxHeight/imgHeight

            #Resize Img
            newWidth = int(imgWidth*ratio)
            newHeight = int(imgHeight*ratio)
            img_original_resized = cv2.resize(img_original, (newWidth,newHeight))

            #Resize skeleton
            for m in meshList:
                m['ver'] *=ratio

            for s in skelList:
                s *=ratio


        else:
            img_original_resized = img_original

        self.renderer.setWindowSize(img_original_resized.shape[1], img_original_resized.shape[0])
        self.renderer.setBackgroundTexture(img_original_resized)
        self.renderer.setViewportSize(img_original_resized.shape[1], img_original_resized.shape[0])

        # self.renderer.add_mesh(meshList[0]['ver'],meshList[0]['f'])
        self.renderer.clear_mesh()
        for mesh in meshList:
            self.renderer.add_mesh(mesh['ver'],mesh['f'])
        self.renderer.showBackground(True)
        self.renderer.setWorldCenterBySceneCenter()
        self.renderer.setCameraViewMode("cam")
        # self.renderer.setViewportSize(img_original_resized.shape[1], img_original_resized.shape[0])
                
        self.renderer.display()
        renderImg = self.renderer.get_screen_color_ibgr()

        if vis:
            viewer2D.ImShow(renderImg,waitTime=1,name="rendered")

        ###Render Side View
        self.renderer.setCameraViewMode("free")     
        self.renderer.setViewAngle(90,20)
        self.renderer.showBackground(False)
        self.renderer.setViewportSize(img_original_resized.shape[1], img_original_resized.shape[0])
        self.renderer.display()
        sideImg = self.renderer.get_screen_color_ibgr()        #Overwite on rawImg

        if vis:
            viewer2D.ImShow(sideImg,waitTime=0,name="sideview")
        
        # sideImg = cv2.resize(sideImg, (renderImg.shape[1], renderImg.shape[0]) )
        self.renderout  ={}
        self.renderout['render_camview'] = renderImg
        self.renderout['render_sideview'] = sideImg


    def visualize_gui_naive(self, meshList, skelList, bboxXYWH_list, img_original):
        """
            args:
                meshList: list of {'ver': pred_vertices, 'f': smpl.faces}
                skelList: list of [JointNum*3, 1]       (where 1 means num. of frames in glviewer)
                bbr_list: list of [x,y,w,h] 
        """
        if len(bboxXYWH_list)>0:
            for bbr in bboxXYWH_list:
                viewer2D.Vis_Bbox(img_original,bbr)
        viewer2D.ImShow(img_original)

        glViewer.setWindowSize(img_original.shape[1], img_original.shape[0])
        # glViewer.setRenderOutputSize(inputImg.shape[1],inputImg.shape[0])
        glViewer.setBackgroundTexture(img_original)
        glViewer.SetOrthoCamera(True)
        glViewer.setMeshData(meshList, bComputeNormal= True)        # meshes = {'ver': pred_vertices, 'f': smplWrapper.f}
        glViewer.setSkeleton(skelList)

        if True:   #Save to File
            if True:        #Cam view rendering
                # glViewer.setSaveFolderName(overlaidImageFolder)
                glViewer.setNearPlane(500)
                glViewer.setWindowSize(img_original.shape[1], img_original.shape[0])
                # glViewer.show_SMPL(bSaveToFile = True, bResetSaveImgCnt = False, countImg = False, mode = 'camera')
                glViewer.show(1)

            if False:    #Side view rendering
                # glViewer.setSaveFolderName(sideImageFolder)
                glViewer.setNearPlane(50)
                glViewer.setWindowSize(img_original.shape[1], img_original.shape[0])
                glViewer.show_SMPL(bSaveToFile = True, bResetSaveImgCnt = False, countImg = True, zoom=1108, mode = 'youtube')
                # glViewer.setSaveFolderName(g_renderDir)
                # glViewer.show(0)
