from __future__ import unicode_literals, print_function
from __future__ import absolute_import

import transforms3d as t3d
import pickle
import chumpy as ch
import numpy as np
from opendr.lighting import LambertianPointLight
from .mano_core.mano_loader import load_model
from .mano_core.lbs import global_rigid_transformation
from .mano_utils import get_keypoints_from_mesh_ch


renderer = None
def pose_hand(mano, K, use_mean_pose=True):
    global renderer
    if renderer is None:
        renderer = HandModel(use_mean_pca=False, use_mean_pose=use_mean_pose)

    # split mano parameters
    poses, shapes, uv_root, scale = split_theta(mano)
    focal, pp = get_focal_pp(K)
    xyz_root = recover_root(uv_root, scale, focal, pp)

    # set up the hand model and feed hand parameters
    renderer.pose_by_root(xyz_root[0], poses[0], shapes[0])
    V, F = renderer._get_verts_faces()
    xyz = renderer._calc_coords()
    return np.array(xyz), np.array(V)


def split_theta(theta):
    poses = theta[:, :48]
    shapes = theta[:, 48:58]
    uv_root = theta[:, 58:60]
    scale = theta[:, 60:]
    return poses, shapes, uv_root, scale

def get_focal_pp(K):
    """ Extract the camera parameters that are relevant for an orthographic assumption. """
    focal = 0.5 * (K[0, 0] + K[1, 1])
    pp = K[:2, 2]
    return focal, pp


def backproject_ortho(uv, scale,  # kind of the predictions
                      focal, pp):  # kind of the camera calibration
    """ Calculate 3D coordinates from 2D coordinates and the camera parameters. """
    uv = uv.copy()
    uv -= pp
    xyz = np.concatenate([np.reshape(uv, [-1, 2]),
                          np.ones_like(uv[:, :1])*focal], 1)
    xyz /= scale
    return xyz


def recover_root(uv_root, scale,
                 focal, pp):
    uv_root = np.reshape(uv_root, [1, 2])
    xyz_root = backproject_ortho(uv_root, scale, focal, pp)
    return xyz_root


class HandModel(object):
    def __init__(self, use_mean_pca=False, use_mean_pose=False):
        MANO_MODEL_PATH = '/Users/rongyu/Documents/research/FAIR/workplace/data/models/smplh_origin/MANO_RIGHT.pkl'
        if use_mean_pca:
            self.model = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=not use_mean_pose,
                                    use_pca=True)
        else:
            self.model = load_model(MANO_MODEL_PATH, ncomps=45, flat_hand_mean=not use_mean_pose,
                                    use_pca=False)

        self.global_trans = ch.array([0.0, 0.0, 0.3])

    def _get_verts_faces(self):
        V = self.model + self.global_trans
        F = self.model.f
        return V, F

    def _calc_coords(self):
        # calculate joint location and rotation
        V, _ = self._get_verts_faces()
        J_regressor = self.model.dd['J_regressor']
        Jtr_x = ch.MatVecMult(J_regressor, V[:, 0])
        Jtr_y = ch.MatVecMult(J_regressor, V[:, 1])
        Jtr_z = ch.MatVecMult(J_regressor, V[:, 2])
        Jtr = ch.vstack([Jtr_x, Jtr_y, Jtr_z]).T
        coords_kp_xyz = get_keypoints_from_mesh_ch(V, Jtr)
        return coords_kp_xyz

    def pose_by_root(self, xyz_root, poses, shapes, root_id=9):
        """ Poses the MANO model according to the root keypoint given. """
        self.model.pose[:] = poses  # set estimated articulation
        self.model.betas[:] = shapes  # set estimated shape
        self.global_trans[:] = 0.0

        # how to chose translation
        xyz = np.array(self._calc_coords())
        global_t = xyz_root - xyz[root_id]  # new - old root keypoint

        # apply changes
        self.global_trans[:] = global_t

    def render(self, cam_intrinsics, dist=None, M=None, img_shape=None, render_mask=False):
        from opendr.camera import ProjectPoints
        from utils.renderer import ColoredRenderer

        if dist is None:
            dist = np.zeros(5)
        dist = dist.flatten()
        if M is None:
            M = np.eye(4)

        # get R, t from M (has to be world2cam)
        R = M[:3, :3]
        ax, angle = t3d.axangles.mat2axangle(R)
        rt = ax*angle
        rt = rt.flatten()
        t = M[:3, 3]

        w, h = (320, 320)
        if img_shape is not None:
            w, h = img_shape[1], img_shape[0]

        pp = np.array([cam_intrinsics[0, 2], cam_intrinsics[1, 2]])
        f = np.array([cam_intrinsics[0, 0], cam_intrinsics[1, 1]])

        # Create OpenDR renderer
        rn = ColoredRenderer()

        # Assign attributes to renderer
        rn.camera = ProjectPoints(rt=rt,
                                  t=t, # camera translation
                                  f=f,  # focal lengths
                                  c=pp,  # camera center (principal point)
                                  k=dist)  # OpenCv distortion params
        rn.frustum = {'near': 0.1, 'far': 5., 'width': w, 'height': h}

        V, F = self._get_verts_faces()

        rn.set(v=V,
               f=F,
               bgcolor=np.zeros(3))

        if render_mask:
            rn.vc = np.ones_like(V)  #for segmentation mask like rendering
        else:
            colors = np.ones_like(V)

            # Construct point light sources
            rn.vc = LambertianPointLight(f=F,
                                         v=V,
                                         num_verts=V.shape[0],
                                         light_pos=np.array([-1000, -1000, -2000]),
                                         vc=0.8 * colors,
                                         light_color=np.array([1., 1., 1.]))

            rn.vc += LambertianPointLight(f=F,
                                          v=V,
                                          num_verts=V.shape[0],
                                          light_pos=np.array([1000, 1000, -2000]),
                                          vc=0.25 * colors,
                                          light_color=np.array([1., 1., 1.]))

            rn.vc += LambertianPointLight(f=F,
                                          v=V,
                                          num_verts=V.shape[0],
                                          light_pos=np.array([2000, 2000, 2000]),
                                          vc=0.1 * colors,
                                          light_color=np.array([1., 1., 1.]))

            rn.vc += LambertianPointLight(f=F,
                                          v=V,
                                          num_verts=V.shape[0],
                                          light_pos=np.array([-2000, -2000, 2000]),
                                          vc=0.1 * colors,
                                          light_color=np.array([1., 1., 1.]))

        # render
        img = (np.array(rn.r) * 255).astype(np.uint8)
        return img

