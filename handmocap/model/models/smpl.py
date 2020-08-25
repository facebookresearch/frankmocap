
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, shutil
import os.path as osp
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_skew(vec, batchSize):
    with torch.cuda.device(vec.get_device()):
        vec = vec.view(batchSize,3)
        res = torch.zeros(batchSize, 9).cuda()
        res[:,1] = -vec[:, 2]
        res[:,2] = vec[:, 1]
        res[:,3] = vec[:, 2]
        res[:,5] = -vec[:, 0]
        res[:,6] = -vec[:, 1]
        res[:,7] = vec[:, 0]
        return res.view(batchSize, 3, 3)



def batch_rodrigues(pose_params):
    with torch.cuda.device(pose_params.get_device()):
        # pose_params shape is (bs*24, 3)
        # angle shape is (batchSize*24, 1)
        angle = torch.norm(pose_params+1e-8, p=2, dim=1).view(-1, 1)
        # r shape is (batchSize*24, 3, 1)
        r = torch.div(pose_params, angle).view(angle.size(0), -1, 1)
        # r_T shape is (batchSize*24, 1, 3)
        r_T = r.permute(0,2,1)
        # cos and sin is (batchSize*24, 1, 1)
        cos = torch.cos(angle).view(angle.size(0), 1, 1)
        sin = torch.sin(angle).view(angle.size(0), 1, 1)
        # outer is (bs*24, 3, 3)
        outer = torch.matmul(r, r_T)
        eye = torch.eye(3).view(1,3,3)
        # eyes is (bs*24, 3, 3)
        eyes = eye.repeat(angle.size(0), 1, 1).cuda()
        # r_sk is (bs*24, 3, 3)
        r_sk = batch_skew(r, r.size(0))
        R = cos * eyes + (1 - cos) * outer + sin * r_sk
        # R shape is (bs*24, 3, 3)
        return R

def make_homo_coords(R, t):
    with torch.cuda.device(R.get_device()):
        bs = R.size(0)
        p1d = (0,0, 0,1)
        R_homo = F.pad(R, p1d, 'constant', 0)
        t_homo = torch.cat([t, torch.ones(bs,1,1).cuda()], dim=1)
        res = torch.cat([R_homo, t_homo], dim=2)
        return res

def batch_rigid_transformation(Rs, Js, parent):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index

    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 4 x 4 relative joint transformations for LBS.
    """
    with torch.cuda.device(Rs.get_device()):
        bs = Rs.size(0)
        root_rotation = Rs[:, 0, :, :]
        # Js now is (bs, 24, 3, 1)
        Js = Js.view(Js.size(0), Js.size(1), Js.size(2), 1)
        # change all the coords to homogeneous representation
        A0 = make_homo_coords(root_rotation, Js[:,0])
        results = [A0]
        for i in range(1, parent.shape[0]):
            j_here = Js[:,i] - Js[:,parent[i]]
            A_here = make_homo_coords(Rs[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)
            results.append(res_here)
        # results (bs, 24, 4, 4)
        results = torch.stack(results, dim=1)
        # new_J is (bs, 24, 3, 1)
        new_J = results[:, :, :3, 3]
        # Js_w0 is (bs, 24, 4, 1)
        Js_w0 = torch.cat([Js, torch.zeros(bs, 24,1,1).cuda()], dim=2)
        # init_bone is (bs, 24, 4, 4)
        init_bone = torch.matmul(results, Js_w0)
        init_bone = F.pad(init_bone, (3,0), 'constant', 0)
        # get final results
        A = results - init_bone
        return new_J, A

def batch_orth_proj_idrot(X, camera):

    # camera is (batchSize, 1, 3)
    camera = camera.view(-1, 1, 3)

    # print("camera", camera[0])
    # x_trans is (batchSize, 19, 2)
    X_trans = X[:, :, :2] + camera[:, :, 1:]

    # first res is (batchSize, 19*2)
    # return value is (batchSize, 19, 2)
    res = camera[:, :, 0] * X_trans.view(X_trans.size(0), -1)
    return res.view(X_trans.size(0), X_trans.size(1), -1)



class SMPL(nn.Module):

    def __init__(self, pkl_path, batchSize):

        super(SMPL, self).__init__()

        with open(pkl_path, 'rb') as in_f:
            data = pickle.load(in_f, encoding='latin1')

        # mean template verts
        # shape of v_template is (1, 6890, 3)
        # shape of self.v_tempalte is (batchSize, 6890, 3)
        v_template = data['v_template']
        self.size = [v_template.shape[0], 3]
        v_template = v_template[np.newaxis, ...]
        v_template = np.repeat(v_template, batchSize, axis=0)
        self.v_template = torch.from_numpy(v_template).float()
        
        # shape of shapedirs is 6890 * 3 * 10
        # shape of self.shapedirs is 10 * 20670 (6890*3)
        # self.beta_num = 10
        shapedirs = data['shapedirs']
        self.beta_num = shapedirs.shape[-1]
        shapedirs = np.transpose(shapedirs.reshape(-1, self.beta_num))
        self.shapedirs = torch.from_numpy(shapedirs).float()

        # J_regressor, shape is 6890 * 24
        J_regressor = data['J_regressor'].T
        self.J_regressor = torch.from_numpy(J_regressor).float()
        
        # shape of posedirs is 6890 * 3 * 207
        # shape of self.posedirs is 207 * 20670
        # self.pose_basis_num = 207
        posedirs = data['posedirs']
        self.pose_basis_num = posedirs.shape[-1]
        posedirs = np.transpose(posedirs.reshape(-1, self.pose_basis_num))
        self.posedirs = torch.from_numpy(posedirs).float()

        # indices of parents for each joints
        # parents = [-1  0  0  0  1  2  3  4  5  6  7  8  9  9  9 12 13 14 16 17 18 19 20 21]
        # len(parents) == 24
        self.parents = data['kintree_table'][0].astype(np.int32)

        # LBS weights, shape is 6890 * 24
        weights = data['weights']
        self.weights = torch.from_numpy(weights).float()

        # 19 keypoints: 6890 * 19
        joint_regressor = data['cocoplus_regressor'].T
        self.joint_regressor = torch.from_numpy(joint_regressor).float()



    def forward(self, shape_params, pose_params, get_skin=True):
        '''
        shape_params: N * 10
        pose_params: N * 72
        '''

        with torch.cuda.device(shape_params.get_device()):
            self.v_template = self.v_template.cuda()
            self.shapedirs = self.shapedirs.cuda()
            self.J_regressor = self.J_regressor.cuda()
            self.posedirs = self.posedirs.cuda()
            self.weights = self.weights.cuda()
            self.joint_regressor = self.joint_regressor.cuda()

            # 1. Add shape blend shapes
            # v_shappe has shape (batchSize, 6890, 3)
            batchSize = shape_params.size(0)
            v_shaped_mid = torch.matmul(shape_params, self.shapedirs)
            v_shaped = v_shaped_mid.view(batchSize, self.size[0], self.size[1]) + self.v_template

            # 2. Infer shape-dependent joint locations
            # Joints (batchSize, 24, 3)
            Jx = torch.matmul(v_shaped[:,:,0], self.J_regressor)
            Jy = torch.matmul(v_shaped[:,:,1], self.J_regressor)
            Jz = torch.matmul(v_shaped[:,:,2], self.J_regressor)
            J = torch.stack([Jx,Jy,Jz], dim=2)

            # 3.Add pose blend shapes
            # change N * 24 * 3 into N * 24 * 3 * 3 (change axis-angle representation into rotation matrix)
            Rs = batch_rodrigues(pose_params.contiguous().view(-1, 3))
            Rs = Rs.view(-1 ,24, 3, 3)
            # Ignore global rotation (not quite understand this process),
            # pose_feature is (bs, 23*3*3=207)
            pose_feature = Rs[:, 1:, :, :] - torch.eye(3).cuda()
            pose_feature = pose_feature.view(-1, 23*3*3)
            # (N * 207) * (207, 20670) ->  N * 6890 * 3
            v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])
            v_posed += v_shaped

            # 4. get the global joint location
            # self.J_transformed is (bs, 24, 3)
            # A is (bs, 24, 4, 4)
            self.J_transformed, A = batch_rigid_transformation(Rs, J, self.parents)


            # 5. Do skinning
            # W is (bs, 6890, 24)
            W = self.weights.repeat(batchSize, 1, 1)
            # T = (bs, 6890, 24) * (bs, 24, 16) -> (bs, 6890, 4, 4)
            T = torch.matmul(W, A.view(batchSize, 24, 16)).view(batchSize, -1, 4, 4)
            # V_posed_homo is (bs, 6890, 4)
            v_posed_homo = torch.cat([v_posed, torch.ones(batchSize, v_posed.shape[1], 1).cuda()], dim=2)
            # V_homo is (bs, 6890, 4, 1)
            v_homo = torch.matmul(T, v_posed_homo.view(batchSize, v_posed_homo.size(1), v_posed_homo.size(2), 1))
            # verts is (bs, 6890, 3)
            verts = v_homo[:, :, :3, 0]

            # Get cocoplus or lsp joints:
            # joinsts is [bs, 19, 3]
            joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
            joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
            joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)
            joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

            if get_skin:
                return verts, joints, self.J_transformed
            else:
                return joints
