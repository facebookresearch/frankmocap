import numpy as np
from transforms3d.quaternions import quat2mat
import sys
sys.path.append('src')

from minimal_hand.config import *
from minimal_hand.kinematics import *
from minimal_hand.utils import *
import pdb
import sys


class HandMesh():
  """
  Wrapper for the MANO hand model.
  """
  def __init__(self, model_path):
    """
    Init.

    Parameters
    ----------
    model_path : str
      Path to the MANO model file. This model is converted by `prepare_mano.py`
      from official release.
    """
    params = load_pkl(model_path)
    self.verts = params['verts']
    self.faces = params['faces']
    self.weights = params['weights']
    self.joints = params['joints']

    self.n_verts = self.verts.shape[0]
    self.n_faces = self.faces.shape[0]

    self.ref_pose = []
    self.ref_T = []
    for j in range(MANOHandJoints.n_joints):
      parent = MANOHandJoints.parents[j]
      if parent is None:
        self.ref_T.append(self.verts)
        self.ref_pose.append(self.joints[j])
      else:
        self.ref_T.append(self.verts - self.joints[parent])
        self.ref_pose.append(self.joints[j] - self.joints[parent])
    self.ref_pose = np.expand_dims(np.stack(self.ref_pose, 0), -1)
    self.ref_T = np.expand_dims(np.stack(self.ref_T, 1), -1)

  def set_abs_quat(self, quat):
    """
    Set absolute (global) rotation for the hand.

    Parameters
    ----------
    quat : np.ndarray, shape [J, 4]
      Absolute rotations for each joint in quaternion.

    Returns
    -------
    np.ndarray, shape [V, 3]
      Mesh vertices after posing.
    """
    mats = []
    for j in range(MANOHandJoints.n_joints):
      mats.append(quat2mat(quat[j]))
    mats = np.stack(mats, 0)

    pose = np.matmul(mats, self.ref_pose)
    joint_xyz = [None] * MANOHandJoints.n_joints
    for j in range(MANOHandJoints.n_joints):
      joint_xyz[j] = pose[j]
      parent = MANOHandJoints.parents[j]
      if parent is not None:
        joint_xyz[j] += joint_xyz[parent]
    joint_xyz = np.stack(joint_xyz, 0)[..., 0]

    T = np.matmul(np.expand_dims(mats, 0), self.ref_T)[..., 0]
    self.verts = [None] * MANOHandJoints.n_joints
    for j in range(MANOHandJoints.n_joints):
      self.verts[j] = T[:, j]
      parent = MANOHandJoints.parents[j]
      if parent is not None:
        self.verts[j] += joint_xyz[parent]
    self.verts = np.stack(self.verts, 1)
    self.verts = np.sum(self.verts * self.weights, 1)

    return self.verts.copy()
