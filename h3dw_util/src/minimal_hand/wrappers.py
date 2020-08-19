import os

import numpy as np
import tensorflow as tf

from minimal_hand.config import *
from minimal_hand.kinematics import *
from minimal_hand.network import *
from minimal_hand.utils import *
import pdb
import sys

class ModelDet:
  """
  DetNet: estimating 3D keypoint positions from input color image.
  """
  def __init__(self, model_path):
    """
    Parameters
    ----------
    model_path : str
      Path to the trained model.
    """
    self.graph = tf.Graph()
    with self.graph.as_default():
      with tf.variable_scope('prior_based_hand'):
        self.sess = tf.Session()
        self.input_ph = tf.placeholder(tf.uint8, [128, 128, 3])
        self.feed_img = \
          tf.cast(tf.expand_dims(self.input_ph, 0), tf.float32) / 255
        self.hmaps, self.dmaps, self.lmaps = \
          detnet(self.feed_img, 1, False)

        self.hmap = self.hmaps[-1]
        self.dmap = self.dmaps[-1]
        self.lmap = self.lmaps[-1]

        self.uv = tf_hmap_to_uv(self.hmap)
        self.delta = tf.gather_nd(
          tf.transpose(self.dmap, [0, 3, 1, 2, 4]), self.uv, batch_dims=2
        )[0]
        self.xyz = tf.gather_nd(
          tf.transpose(self.lmap, [0, 3, 1, 2, 4]), self.uv, batch_dims=2
        )[0]

        self.uv = self.uv[0]
      tf.train.Saver().restore(self.sess, model_path)

  def process(self, img):
    """
    Process a color image.

    Parameters
    ----------
    img : np.ndarray
      A 128x128 RGB image of **left hand** with dtype uint8.

    Returns
    -------
    np.ndarray, shape [21, 3]
      Normalized keypoint locations. The coordinates are relative to the M0
      joint and normalized by the length of the bone from wrist to M0. The
      order of keypoints is as `kinematics.MPIIHandJoints`.
    np.ndarray, shape [21, 2]
      The uv coordinates of the keypoints on the heat map, whose resolution is
      32x32.
    """
    results = self.sess.run([self.xyz, self.uv], {self.input_ph: img})
    return results


class ModelIK:
  """
  IKnet: estimating joint rotations from locations.
  """
  def __init__(self, input_size, network_fn, model_path, net_depth, net_width):
    """
    Parameters
    ----------
    input_size : int
      Number of joints to be used, e.g. 21, 42.
    network_fn : function
      Network function from `network.py`.
    model_path : str
      Path to the trained model.
    net_depth : int
      Number of layers.
    net_width : int
      Number of neurons in each layer.
    """
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.input_ph = tf.placeholder(tf.float32, [1, input_size, 3])
      with tf.name_scope('network'):
        self.theta = \
          network_fn(self.input_ph, net_depth, net_width, training=False)[0]
      self.sess = tf.Session()
      tf.train.Saver().restore(self.sess, model_path)

  def process(self, joints):
    """
    Estimate joint rotations from locations.

    Parameters
    ----------
    joints : np.ndarray, shape [N, 3]
      Input joint locations (and other information e.g. bone orientation).

    Returns
    -------
    np.ndarray, shape [21, 4]
      Estimated global joint rotations in quaternions.
    """
    theta = \
      self.sess.run(self.theta, {self.input_ph: np.expand_dims(joints, 0)})
    if len(theta.shape) == 3:
      theta = theta[0]
    return theta


class ModelPipeline:
  """
  A wrapper that puts DetNet and IKNet together.
  """
  def __init__(self):
    # load reference MANO hand pose
    mano_ref_xyz = load_pkl(HAND_MESH_MODEL_PATH)['joints']
    # convert the kinematic definition to MPII style, and normalize it
    mpii_ref_xyz = mano_to_mpii(mano_ref_xyz) / IK_UNIT_LENGTH
    mpii_ref_xyz -= mpii_ref_xyz[9:10]
    # get bone orientations in the reference pose
    mpii_ref_delta, mpii_ref_length = xyz_to_delta(mpii_ref_xyz, MPIIHandJoints)
    mpii_ref_delta = mpii_ref_delta * mpii_ref_length

    self.mpii_ref_xyz = mpii_ref_xyz
    self.mpii_ref_delta = mpii_ref_delta

    self.det_model = ModelDet(DETECTION_MODEL_PATH)
    # 84 = 21 joint coordinates
    #    + 21 bone orientations
    #    + 21 joint coordinates in reference pose
    #    + 21 bone orientations in reference pose
    self.ik_model = ModelIK(84, iknet, IK_MODEL_PATH, 6, 1024)

  def process(self, frame):
    """
    Process a single frame.

    Parameters
    ----------
    frame : np.ndarray, shape [128, 128, 3], dtype np.uint8.
      Frame to be processed.

    Returns
    -------
    np.ndarray, shape [21, 3]
      Joint locations.
    np.ndarray, shape [21, 4]
      Joint rotations.
    """
    xyz, _ = self.det_model.process(frame)
    delta, length = xyz_to_delta(xyz, MPIIHandJoints)
    delta *= length
    pack = np.concatenate(
      [xyz, delta, self.mpii_ref_xyz, self.mpii_ref_delta], 0
    )
    theta = self.ik_model.process(pack)

    return xyz, theta
  

class ModelConverter:
  """
  A wrapper that puts DetNet and IKNet together.
  """
  def __init__(self):
    # load reference MANO hand pose
    mano_ref_xyz = load_pkl(HAND_MESH_MODEL_PATH)['joints']
    # convert the kinematic definition to MPII style, and normalize it
    mpii_ref_xyz = mano_to_mpii(mano_ref_xyz) / IK_UNIT_LENGTH
    mpii_ref_xyz -= mpii_ref_xyz[9:10]
    # get bone orientations in the reference pose
    mpii_ref_delta, mpii_ref_length = xyz_to_delta(mpii_ref_xyz, MPIIHandJoints)
    mpii_ref_delta = mpii_ref_delta * mpii_ref_length

    self.mpii_ref_xyz = mpii_ref_xyz
    self.mpii_ref_delta = mpii_ref_delta

    self.det_model = ModelDet(DETECTION_MODEL_PATH)
    # 84 = 21 joint coordinates
    #    + 21 bone orientations
    #    + 21 joint coordinates in reference pose
    #    + 21 bone orientations in reference pose
    self.ik_model = ModelIK(84, iknet, IK_MODEL_PATH, 6, 1024)

  def process(self, joints_3d):
    """
    Process a single frame.

    Parameters
    ----------
    frame : np.ndarray, shape [128, 128, 3], dtype np.uint8.
      Frame to be processed.

    Returns
    -------
    np.ndarray, shape [21, 3]
      Joint locations.
    np.ndarray, shape [21, 4]
      Joint rotations.
    """
    delta, length = xyz_to_delta(joints_3d, MPIIHandJoints)
    delta *= length
    pack = np.concatenate(
      [joints_3d, delta, self.mpii_ref_xyz, self.mpii_ref_delta], 0
    )
    theta = self.ik_model.process(pack)
    return theta
