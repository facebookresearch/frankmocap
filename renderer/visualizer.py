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
from renderer import glViewer
from renderer import meshRenderer #glRenderer

from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm

from renderer.image_utils import draw_raw_bbox, draw_hand_bbox, draw_body_bbox, draw_arm_pose

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
        rendererType ='opengl_gui'          #nongui or gui
    ):
        self.rendererType = rendererType
        if rendererType != "opengl_gui" and rendererType!= "opengl":
            print("Wrong rendererType: {rendererType}")
            assert False

        self.cam_all = []
        self.vert_all = []
        self.bboxXYXY_all = []

        self.bg_image = None

        #Screenless rendering
        if rendererType =='opengl':
            self.renderer = meshRenderer.meshRenderer()
            self.renderer.setRenderMode('geo')
            self.renderer.offscreenMode(True)
        else:
            from renderer import glViewer #glRenderer
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

    def visualize(self,
        input_img, 
        hand_bbox_list = None, 
        body_bbox_list = None,
        body_pose_list = None,
        raw_hand_bboxes = None,
        pred_mesh_list = None,
        vis_raw_hand_bbox = True,
        vis_body_pose = True,
        vis_hand_bbox = True,
    ):
         # init
        res_img = input_img.copy()

        # draw raw hand bboxes
        if raw_hand_bboxes is not None and vis_raw_hand_bbox:
            res_img = draw_raw_bbox(input_img, raw_hand_bboxes)
            # res_img = np.concatenate((res_img, raw_bbox_img), axis=1)

        # draw 2D Pose
        if body_pose_list is not None and vis_body_pose:
            res_img = draw_arm_pose(res_img, body_pose_list)

        # draw body bbox
        if body_bbox_list is not None:
            body_bbox_img = draw_body_bbox(input_img, body_bbox_list)
            res_img = body_bbox_img

        # draw hand bbox
        if hand_bbox_list is not None:
            res_img = draw_hand_bbox(res_img, hand_bbox_list)

        # render predicted meshes
        if pred_mesh_list is not None:
            rend_img = self.__render_pred_verts(input_img, pred_mesh_list)
            if rend_img is not None:
                res_img = np.concatenate((res_img, rend_img), axis=1)
            # res_img = rend_img
        
        return res_img
        
    def __render_pred_verts(self, img_original, pred_mesh_list):

        res_img = img_original.copy()

        pred_mesh_list_offset =[]
        for mesh in pred_mesh_list:

            # Mesh vertices have in image cooridnate (left, top origin)
            # Move the X-Y origin in image center
            mesh_offset = mesh['vertices'].copy()
            mesh_offset[:,0] -= img_original.shape[1]*0.5
            mesh_offset[:,1] -= img_original.shape[0]*0.5
            pred_mesh_list_offset.append( {'ver': mesh_offset, 'f':mesh['faces'] })# verts = mesh['vertices']
            # faces = mesh['faces']
        if self.rendererType =="opengl_gui":
            self._visualize_gui_naive(pred_mesh_list_offset, img_original=res_img)
            overlaidImg = None
        else:
            self._visualize_screenless_naive(pred_mesh_list_offset, img_original=res_img)
            overlaidImg = self.renderout['render_camview']
            # sideImg = self.renderout['render_sideview']

        return overlaidImg


    def _visualize_screenless_naive(self, meshList, skelList=None, body_bbox_list=None, img_original=None, show_side = False, vis=False, maxHeight = 1080):
        
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
        
        if body_bbox_list is not None:
            for bbr in body_bbox_list:
                viewer2D.Vis_Bbox(img_original,bbr)
        # viewer2D.ImShow(img_original)

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

            if skelList is not None:
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
        if show_side:
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

        if show_side:
            self.renderout['render_sideview'] = sideImg


    def _visualize_gui_naive(self, meshList, skelList=None, body_bbox_list=None, img_original=None, normal_compute=True):
        """
            args:
                meshList: list of {'ver': pred_vertices, 'f': smpl.faces}
                skelList: list of [JointNum*3, 1]       (where 1 means num. of frames in glviewer)
                bbr_list: list of [x,y,w,h] 
        """
        if body_bbox_list is not None:
            for bbr in body_bbox_list:
                viewer2D.Vis_Bbox(img_original, bbr)
        # viewer2D.ImShow(img_original)

        glViewer.setWindowSize(img_original.shape[1], img_original.shape[0])
        # glViewer.setRenderOutputSize(inputImg.shape[1],inputImg.shape[0])
        glViewer.setBackgroundTexture(img_original)
        glViewer.SetOrthoCamera(True)
        glViewer.setMeshData(meshList, bComputeNormal= normal_compute)        # meshes = {'ver': pred_vertices, 'f': smplWrapper.f}

        if skelList is not None:
            glViewer.setSkeleton(skelList)

        if True:   #Save to File
            if True:        #Cam view rendering
                # glViewer.setSaveFolderName(overlaidImageFolder)
                glViewer.setNearPlane(50)
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
    

    def _visualize_gui_smplpose_basic(self, smpl, poseParamList, shapeParamList =None,  colorList = None, isRotMat = False, scalingFactor=300, waittime =1):
        '''
            Visualize SMPL vertices from SMPL pose parameters
            This can be used as a quick visualize function if you have a pose parameters

            args: 
                poseParamList: list of pose parameters (numpy array) in angle axis (72,) by default  or rot matrix (24,3,3) with isRotMat==True
                shapeParamList: (optional) list of shape parameters (numpy array) (10,). If not provided, use a zero vector
                colorList: (optional) list of color RGB values e.g., (255,0,0) for red
        '''
        zero_betas = torch.from_numpy(np.zeros( (1,10), dtype=np.float32))
        default_color = glViewer.g_colorSet['eft']
        meshList =[]
        for i, poseParam in enumerate(poseParamList):
            
            if shapeParamList is not None:
                shapeParam = torch.from_numpy(shapeParamList[i][np.newaxis,:])
            else:
                shapeParam = zero_betas#.copy()

            if colorList is not None:
                color = colorList[i]
            else:
                color = default_color

            poseParam_tensor = torch.from_numpy( poseParam[np.newaxis,:]).float()
            if isRotMat:        #rot mat
                pred_output = smpl(betas=shapeParam, body_pose=poseParam_tensor[:,1:], global_orient=poseParam_tensor[:,[0]], pose2rot=False)

            else:  #angle axis
                pred_output = smpl(betas=shapeParam, body_pose=poseParam_tensor[:,3:], global_orient=poseParam_tensor[:,:3], pose2rot=True)
        
            nn_vertices = pred_output.vertices.detach()[0].numpy() * scalingFactor
            tempMesh = {'ver': nn_vertices, 'f':  smpl.faces, 'color':color}
            meshList.append(tempMesh)
            
        
        # glViewer.setRenderOutputSize(inputImg.shape[1],inputImg.shape[0])
        # glViewer.setBackgroundTexture(img_original)
        glViewer.g_bShowFloor = True
        # glViewer.SetOrthoCamera(True)
        glViewer.g_viewMode = 'free'
        glViewer.setMeshData(meshList, bComputeNormal= True)        # meshes = {'ver': pred_vertices, 'f': smplWrapper.f}
        # glViewer.PuttingObjectCenter()
        # glViewer.setSkeleton(skelList)
        glViewer.show(waittime)        #Press q to escape
