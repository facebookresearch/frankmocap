import numpy as np


class MANOHandJoints:
  n_joints = 21

  labels = [
    'W', #0
    'I0', 'I1', 'I2', #3
    'M0', 'M1', 'M2', #6
    'L0', 'L1', 'L2', #9
    'R0', 'R1', 'R2', #12
    'T0', 'T1', 'T2', #15
    'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
  ]

  # finger tips are not joints in MANO, we label them on the mesh manually
  mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}

  parents = [
    None,
    0, 1, 2,
    0, 4, 5,
    0, 7, 8,
    0, 10, 11,
    0, 13, 14,
    3, 6, 9, 12, 15
  ]


class MPIIHandJoints:
  n_joints = 21

  labels = [
    'W', #0
    'T0', 'T1', 'T2', 'T3', #4
    'I0', 'I1', 'I2', 'I3', #8
    'M0', 'M1', 'M2', 'M3', #12
    'R0', 'R1', 'R2', 'R3', #16
    'L0', 'L1', 'L2', 'L3', #20
  ]

  parents = [
    None,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19
  ]


def mpii_to_mano(mpii):
  """
  Map data from MPIIHandJoints order to MANOHandJoints order.

  Parameters
  ----------
  mpii : np.ndarray, [21, ...]
    Data in MPIIHandJoints order. Note that the joints are along axis 0.

  Returns
  -------
  np.ndarray
    Data in MANOHandJoints order.
  """
  mano = []
  for j in range(MANOHandJoints.n_joints):
    mano.append(
      mpii[MPIIHandJoints.labels.index(MANOHandJoints.labels[j])]
    )
  mano = np.stack(mano, 0)
  return mano


def mano_to_mpii(mano):
  """
  Map data from MANOHandJoints order to MPIIHandJoints order.

  Parameters
  ----------
  mano : np.ndarray, [21, ...]
    Data in MANOHandJoints order. Note that the joints are along axis 0.

  Returns
  -------
  np.ndarray
    Data in MPIIHandJoints order.
  """
  mpii = []
  for j in range(MPIIHandJoints.n_joints):
    mpii.append(
      mano[MANOHandJoints.labels.index(MPIIHandJoints.labels[j])]
    )
  mpii = np.stack(mpii, 0)
  return mpii


def xyz_to_delta(xyz, joints_def):
  """
  Compute bone orientations from joint coordinates (child joint - parent joint).
  The returned vectors are normalized.
  For the root joint, it will be a zero vector.

  Parameters
  ----------
  xyz : np.ndarray, shape [J, 3]
    Joint coordinates.
  joints_def : object
    An object that defines the kinematic skeleton, e.g. MPIIHandJoints.

  Returns
  -------
  np.ndarray, shape [J, 3]
    The **unit** vectors from each child joint to its parent joint.
    For the root joint, it's are zero vector.
  np.ndarray, shape [J, 1]
    The length of each bone (from child joint to parent joint).
    For the root joint, it's zero.
  """
  delta = []
  for j in range(joints_def.n_joints):
    p = joints_def.parents[j]
    if p is None:
      delta.append(np.zeros(3))
    else:
      delta.append(xyz[j] - xyz[p])
  delta = np.stack(delta, 0)
  lengths = np.linalg.norm(delta, axis=-1, keepdims=True)
  delta /= np.maximum(lengths, np.finfo(xyz.dtype).eps)
  return delta, lengths
