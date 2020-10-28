import os
import pickle
import cv2
import torch
import numpy as np

# from eft.eftSingleView import EFTFitterHMR, process_image
# from fairmocap.utils.geometry import weakProjection_gpu
# from renderer import viewer2D,glViewer
# from fairmocap.utils.imutils import convert_smpl_to_bbox, convert_bbox_to_oriIm

from torchvision.transforms import Normalize
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm

#The following is for EFT
import copy
import torch
import torch.nn as nn

# from mocap_utils.geometry_utils import rotmat3x3_to_angleaxis
import mocap_utils.geometry_utils as gu
from bodymocap.utils.geometry import weakProjection_gpu

import renderer.glViewer as glViewer
import renderer.viewer2D as viewer2D

class Whole_Body_eft():
    """
    This EFT code is for in-the-wild data where no GT is available
    """

    def __init__(self, model, smpl):
        #Basic
        self.smpl_mapping = pickle.load(open("/home/hjoo/codes/handmocap/SMPLX_HAND_INFO.pkl", "rb"))
        
        self.model_regressor = model

        lr = 5e-5 * 0.2
        self.optimizer = torch.optim.Adam(params=self.model_regressor.parameters(),
                                        #   lr=self.options.lr,
                                            lr =lr,
                                          weight_decay=0)

        self.backupModel()
        self.smpl = smpl

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Define criteria
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)

        #debug
        from torchvision.transforms import Normalize
        IMG_RES = 224

        # Mean and standard deviation for normalizing input image
        IMG_NORM_MEAN = [0.485, 0.456, 0.406]
        IMG_NORM_STD = [0.229, 0.224, 0.225]
        self.de_normalize_img = Normalize(mean=[ -IMG_NORM_MEAN[0]/IMG_NORM_STD[0]    , -IMG_NORM_MEAN[1]/IMG_NORM_STD[1], -IMG_NORM_MEAN[2]/IMG_NORM_STD[2]], std=[1/IMG_NORM_STD[0], 1/IMG_NORM_STD[1], 1/IMG_NORM_STD[2]])


        self.beta_loss_weight = 0.001
        self.keypoint_loss_weight = 5.


    # EFT ##########################
    def backupModel(self):
        print(">>> Model status saved!")
        self.model_backup = copy.deepcopy(self.model_regressor.state_dict())
        # self.optimizer_backup = copy.deepcopy(self.optimizer.state_dict())

    def reloadModel(self):
        print(">>> Model status has been reloaded to initial!")
        self.model_regressor.load_state_dict(self.model_backup)
        # self.optimizer.load_state_dict(self.optimizer_backup)

        lr = 5e-5 * 0.2
        self.optimizer = torch.optim.Adam(params=self.model_regressor.parameters(),
                                        #   lr=self.options.lr,
                                            lr =lr,
                                          weight_decay=0)
    def exemplerTrainingMode(self):

        for module in self.model_regressor.modules():
            if type(module)==False:
                continue

            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                # print(module)
                module.eval()
                for m in module.parameters():
                    m.requires_grad =False
            if isinstance(module, nn.Dropout):
                # print(module)
                module.eval()
                for m in module.parameters():
                    m.requires_grad =False
  
    def compute_loss_straight_legs(self, pred_rotmat):
        loss_leg = pred_rotmat[:,[4,5],:,:] - torch.from_numpy(np.eye(3)).float().cuda()     #0 is root
        loss_leg = torch.mean(loss_leg**2)

        return loss_leg

    def compute_loss_torso_upright(self, pred_rotmat):
        """
        Make sure that the origin and torso are upright
        """      

        matupright = np.eye(3)
        matupright[1,1] =-1
        matupright[2,2] =-1
        loss_origin_upright = pred_rotmat[:,[0],:,:] - torch.from_numpy( matupright ).float().cuda()         

        # loss_torso_spine = pred_rotmat[:,[1,2 ,3,6,9],:,:] - torch.from_numpy(np.eye(3)).float().cuda()     
        loss_torso_spine = pred_rotmat[:,[1,2 ],:,:] - torch.from_numpy(np.eye(3)).float().cuda()     #1,2: hips 3,6,9:  spine 1,2,3
        loss_torso_upright = torch.mean(loss_origin_upright**2) + torch.mean(loss_torso_spine**2)


        return loss_torso_upright

    def keypoint_loss_openpose25(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.

        Note that the order is in openpose ordering
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d[:,:25,:], gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_loss_keypoint21(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf *= openpose_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def compute_loss(self, input_batch, pred_rotmat, pred_betas, pred_camera,
                    bNoHand= False, bNoLegs= False):
        if bNoHand==False:#g_bUse3DForOptimization:

            right_hand_pose = None
            left_hand_pose = None
            if 'rhand_pose' in input_batch: 
                right_hand_pose = input_batch['rhand_pose']
            
            if 'lhand_pose' in input_batch: 
                left_hand_pose = input_batch['lhand_pose']

            # pred_aa = rotmat3x3_to_angleaxis(pred_rotmat)
            pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).cuda()
            pred_aa = pred_aa.view(pred_aa.shape[0],-1)
          
            # pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,[0]], pose2rot=False, right_hand_pose= right_hand_pose, left_hand_pose= left_hand_pose)
            pred_output = self.smpl(
                    betas=pred_betas, 
                    body_pose=pred_aa[:,3:], 
                    global_orient=pred_aa[:,:3], 
                    pose2rot=True,
                    right_hand_pose= right_hand_pose, left_hand_pose= left_hand_pose
                    )
        else:        #No Hand
            assert False
            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,[0]], pose2rot=False)

        pred_vertices = pred_output.vertices
        pred_joints_3d = pred_output.joints
        pred_keypoints_2d = weakProjection_gpu(pred_joints_3d, pred_camera[:,0], pred_camera[:,1:] )           #N, 49, 2
        pred_right_hand_2d = weakProjection_gpu(pred_output.right_hand_joints, pred_camera[:,0], pred_camera[:,1:] )           #N, 49, 2
        pred_left_hand_2d = weakProjection_gpu(pred_output.left_hand_joints, pred_camera[:,0], pred_camera[:,1:] )           #N, 49, 2
        

        #Get GT data 
        gt_keypoints_2d = input_batch['body_joints_2d_bboxNormCoord']# 2D keypoints           #[N,49,3]  or [N,25,3]
        gt_rhand_2d = input_batch['rhand_joints_2d_bboxNormCoord']# 2D keypoints           #[N,49,3]  or [N,25,3]
        gt_lhand_2d = input_batch['lhand_joints_2d_bboxNormCoord']# 2D keypoints           #[N,49,3]  or [N,25,3]
        init_rotmat = input_batch['init_rotmat']# 2D keypoints           #[N,49,3]  or [N,25,3]

        loss_keypoints_2d = self.keypoint_loss_openpose25(pred_keypoints_2d, gt_keypoints_2d,1.0)
                                            #self.options.openpose_train_weight)

        loss_keypoints_2d_hand = self.keypoint_loss_keypoint21(pred_right_hand_2d, gt_rhand_2d,1.0) + self.keypoint_loss_keypoint21(pred_left_hand_2d, gt_lhand_2d,1.0)


        loss_pose_3d = self.criterion_keypoints(init_rotmat, pred_rotmat).mean()         #pred_rotmat [N,24,3,3]
        loss_regr_betas_noReject = torch.mean(pred_betas**2)

        #Standing pose
        loss_straight_legs = self.compute_loss_straight_legs(pred_rotmat)
        loss_torso_upright = self.compute_loss_torso_upright(pred_rotmat)
     
        #Add hand keypoint loss
        if True:
            pred_wrists={}
            # camParam_scale = pred_camera[b,0]
            # camParam_trans = pred_camera[b,1:]
            # pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)

            pred_joints_bbox =  pred_joints_3d* pred_camera[:,0]
            # pred_joints_vis *= pred_camera_vis[b,0]
            pred_joints_bbox[:,:, :2] = pred_joints_bbox[:,:, :2] + pred_camera[:,1:].unsqueeze(1)
            pred_joints_bbox *=112           #112 == 0.5*224

            
            pred_vertices_bbox =  pred_vertices* pred_camera[:,0]
            # pred_joints_vis *= pred_camera_vis[b,0]
            pred_vertices_bbox[:,:, :2] = pred_vertices_bbox[:,:, :2] + pred_camera[:,1:].unsqueeze(1)
            pred_vertices_bbox *=112           #112 == 0.5*224

            #Save wrist joint to localize 3D hand mesh in the current wrist location
            pred_wrists['r'] = pred_joints_bbox[:,25+6]   #[N,49,3]
            pred_wrists['l'] = pred_joints_bbox[:,25+11]   #[N,49,3]

            #compute hand mesh loss
            handMeshVert ={}
            handMeshVert['l'] = None
            handMeshVert['r'] = None
            if True:
                hadMeshLossAll=torch.tensor(0.0).to(self.device)
                for lr in ["l", "r"]:
                    
                    if f'{lr}hand_joints' not in input_batch or f'{lr}hand_pose' not in input_batch:
                        continue
                    
                    # camParam_scale = pred_camera_vis[b,0]
                    scaleFactor = 1.0 #bboxScale_o2n
                    # input_batch['handData'][f'{lr}_mesh']['ver']         #778,3
                    hand_wristJoint = input_batch[f'{lr}hand_joints'][[0],:] * scaleFactor             #21,3
                    deltaWrist = pred_wrists[lr] - hand_wristJoint

                    #Get Hand mesh
                    if lr =='r':
                        vertId_map = np.array(self.smpl_mapping['right_hand_verts_idx']) #vertId_map[localId] = SMPLX-ID
                    else:
                        vertId_map = np.array(self.smpl_mapping['left_hand_verts_idx']) #vertId_map[localId] = SMPLX-ID

                    hand_mesh =  input_batch[f'{lr}hand_verts']*scaleFactor       #778,3
                    hand_mesh =  hand_mesh+ deltaWrist
                    handMeshVert[lr] = {'ver':hand_mesh.clone().detach().cpu().numpy(), 'f':input_batch[f'{lr}hand_faces'], 'color':(50,200,50)}       #Save as numpy for debugging


                    if False:
                        hand_mesh_extend = torch.from_numpy(np.zeros(pred_vertices_bbox.shape, dtype=np.float32)).to(self.device)
                        hand_mesh_extend[0,vertId_map,:] = hand_mesh
                        # hand_mesh_extend = torch.from_numpy(hand_mesh_extend).to(self.device)

                        # hand_mesh_extend = hand_mesh_extend + deltaWrist


                        
                        hand_mesh_extend_conf = np.zeros(pred_vertices_bbox.shape, dtype=np.float32)
                        hand_mesh_extend_conf[0,vertId_map,:] = 1.0
                        hand_mesh_extend_conf = torch.from_numpy(hand_mesh_extend_conf).to(self.device)
                        #Find the mapping between hand_mesh to smpl's hand
                        handMeshloss = (hand_mesh_extend_conf*self.criterion_keypoints(pred_vertices_bbox, hand_mesh_extend)).mean() * (10475/ 778)
                    else:
                        handMeshloss = self.criterion_keypoints(pred_vertices_bbox[0,vertId_map,:], hand_mesh).mean() * (10475/ 778)

                    hadMeshLossAll += handMeshloss * 10
        
        ############# Define losses  #############
        #Default: prior loss
        loss = ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean() +  self.beta_loss_weight * loss_regr_betas_noReject

        #Torso orientation
        if bNoLegs:
            if True:
                loss = loss + loss_torso_upright
            #No knee motion
            if True:
                loss = loss + loss_straight_legs

        if True:
            loss = loss + self.keypoint_loss_weight * loss_keypoints_2d * 50
        
        if True:        ##TODO: if 3D hand is not valid, sometimes this doesn't work well
            loss = loss + self.keypoint_loss_weight * loss_keypoints_2d_hand *50
        #         
        
        # loss = loss_pose_3d + \
        if bNoHand==False:#g_bUse3DForOptimization:
            loss = loss + hadMeshLossAll*1e-4
        
        return loss, pred_output, handMeshVert
            

    def vis_eft_step_output(self, input_batch, pred_camera, pred_smpl_output, hand_mesh):

        curImgVis = input_batch['img_cropped_rgb'].copy()     #3,224,224
        #Denormalize image
        viewer2D.ImShow(curImgVis,waitTime=1)

        ############### Visualize Mesh ############### 
        pred_vert_vis = pred_smpl_output.vertices.detach().cpu().numpy()[0]
        pred_camera = pred_camera.detach().cpu().numpy().ravel()
        camScale = pred_camera[0] # *1.15
        camTrans = pred_camera[1:]
        pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camScale, camTrans)
        # pred_vert_vis *=pred_camera_vis[b,0]
        # pred_vert_vis[:,0] += pred_camera_vis[b,1]        #no need +1 (or  112). Rendernig has this offset already
        # pred_vert_vis[:,1] += pred_camera_vis[b,2]        #no need +1 (or  112). Rendernig has this offset already
        # pred_vert_vis*=112
        pred_meshes = {'ver': pred_vert_vis, 'f': self.smpl.faces}

        # # input_batch['handData']['right_mesh']
        mesh_list = [pred_meshes]
        # mesh_list =[]       #Debug


            # ############### Visualize Skeletons ############### 
            # # #Vis pred-SMPL joint
            # pred_joints_bbox_vis = pred_joints_bbox[b,:,:3].detach().cpu().numpy()  #[N,49,3]
            # pred_joints_bbox_vis = pred_joints_bbox_vis.ravel()[:,np.newaxis]
            # glViewer.setSkeleton( [pred_joints_bbox_vis], jointType='spin')

          
            # # #Add hand mesh 
            # # #shift size
            # if bNoHand==False:
            #     for rl in ["r", "l"] :
            #         # scaleFactor = 1.0#bboxScale_o2n
            #         # print(bboxScale_o2n)
            #         # input_batch['handData'][f'{lr}_mesh']['ver']         #778,3
            #         # hand_wristJoint = input_batch['handData'][f'{lr}_joint'][:1,:] * scaleFactor             #21,3

            #         # delta = pred_wrists[lr].T - hand_wristJoint
            #         # print(delta)
            #         if f'{rl}hand_faces' not in input_batch:
            #             continue

            #         mesh_face= input_batch[f'{rl}hand_faces']
            #         tempHandMesh={'ver': handMeshVert[rl].copy(), 'f':mesh_face ,'color':(30, 178, 166) }
            #         mesh_list.append(tempHandMesh)

            # # # glViewer.setMeshData([pred_meshes], bComputeNormal= True)
        glViewer.setMeshData(mesh_list, bComputeNormal= True)
        glViewer.setBackgroundTexture(curImgVis)       #Vis raw video as background
        glViewer.setWindowSize(curImgVis.shape[1]*4, curImgVis.shape[0]*4)
        glViewer.SetOrthoCamera(True)



        #add hand mesh 
        for lr in ['l', 'r']:
            if hand_mesh[lr] is not None:
                glViewer.addMeshData([hand_mesh[lr]], bComputeNormal= True)

        glViewer.show(0)


    def eft_run(self, input_batch, eftIterNum = 20, is_vis= False):

        self.reloadModel()
        
        self.model_regressor.train()
        self.exemplerTrainingMode()
        # self.model_regressor.train()
        self.exemplerTrainingMode()

        # input_batch['img_norm'] = input_batch['img_norm'].to(self.device) # input image
        # input_batch['keypoints_2d'] = input_batch['keypoints_2d'].to(self.device) # input image

        for _ in range(eftIterNum):
            pred_rotmat, pred_betas, pred_camera  = self.eftStep(input_batch, is_vis= is_vis)

        #Reset Model
        self.reloadModel()

        return pred_rotmat, pred_betas, pred_camera


    def eftStep(self, input_batch, is_vis=False, bNoHand = False, bNoLegs = True, visDebugPause= True):

        # Get data from the batch
        images = input_batch['img'].to(self.device) # input image
        batch_size = images.shape[0]

        #Run prediction
        pred_rotmat, pred_betas, pred_camera = self.model_regressor(images)

        loss, pred_smpl_output, hand_mesh_debug = self.compute_loss(input_batch, pred_rotmat, pred_betas, pred_camera)
        print(f"EFTstep>> loss: {loss}")

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if is_vis:#self.options.bDebug_visEFT:#g_debugVisualize:    #Debug Visualize input
            self.vis_eft_step_output(input_batch, pred_camera, pred_smpl_output, hand_mesh_debug)

        return pred_rotmat, pred_betas, pred_camera #, losses

       # #For all sample in the current trainingDB