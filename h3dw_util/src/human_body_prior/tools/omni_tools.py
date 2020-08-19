# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.01.02
import numpy as np

copy2cpu = lambda tensor: tensor.detach().cpu().numpy()

colors = {
    'pink': [.7, .7, .9],
    'purple': [.9, .7, .7],
    'cyan': [.7, .75, .5],
    'red': [1.0, 0.0, 0.0],

    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [.5, .7, .7],
    'blue': [.0, .0, 1.],

    'offwhite': [.8, .9, .9],
    'white': [1., 1., 1.],
    'orange': [.5, .65, .9],

    'grey': [.7, .7, .7],
    'black': np.zeros(3),
    'white': np.ones(3),

    'yellowg': [0.83, 1, 0],
}

def id_generator(size=13):
    import string
    import random
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

def log2file(logpath=None, auto_newline = True):
    import sys
    if logpath is not None:
        makepath(logpath, isfile=True)
        fhandle = open(logpath,'a+')
    else:
        fhandle = None
    def _(text):
        if text is None: return
        if auto_newline:
            if not text.endswith('\n'):
                text = text + '\n'
        sys.stderr.write(text)
        if fhandle is not None:
            fhandle.write(text)
            fhandle.flush()

    return lambda text: _(text)

def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

def matrot2axisangle(matrots):
    '''
    :param matrots: N*T*num_joints*9
    :return: N*T*num_joints*3
    '''
    import cv2
    N = matrots.shape[0]
    T = matrots.shape[1]
    n_joints = matrots.shape[2]
    out_axisangle = []
    for tIdx in range(T):
        T_axisangle = []
        for mIdx in range(N):
            cur_axisangle = []
            for jIdx in range(n_joints):
                cur_axisangle.append(cv2.Rodrigues(matrots[mIdx, tIdx, jIdx:jIdx + 1, :].reshape(3, 3))[0].T)
            T_axisangle.append(np.vstack(cur_axisangle)[np.newaxis])
        out_axisangle.append(np.vstack(T_axisangle).reshape([N,1, -1,3]))
    return np.concatenate(out_axisangle, axis=1)

def axisangle2matrots(axisangle):
    '''
    :param matrots: N*1*num_joints*3
    :return: N*num_joints*9
    '''
    import cv2
    batch_size = axisangle.shape[0]
    axisangle = axisangle.reshape([batch_size,1,-1,3])
    out_matrot = []
    for mIdx in range(axisangle.shape[0]):
        cur_axisangle = []
        for jIdx in range(axisangle.shape[2]):
            a = cv2.Rodrigues(axisangle[mIdx, 0, jIdx:jIdx + 1, :].reshape(1, 3))[0].T
            cur_axisangle.append(a)

        out_matrot.append(np.array(cur_axisangle).reshape([batch_size,1,-1,9]))
    return np.vstack(out_matrot)

def em2euler(em):
    '''

    :param em: rotation in expo-map (3,)
    :return: rotation in euler angles (3,)
    '''
    from transforms3d.euler import axangle2euler

    theta = np.sqrt((em ** 2).sum())
    axis = em / theta
    return np.array(axangle2euler(axis, theta))

def euler2em(ea):
    '''

    :param ea: rotation in euler angles (3,)
    :return: rotation in expo-map (3,)
    '''
    from transforms3d.euler import euler2axangle
    axis, theta = euler2axangle(*ea)
    return np.array(axis*theta)

def apply_mesh_tranfsormations_(meshes, transf):
    '''
    apply inplace translations to meshes
    :param meshes: list of trimesh meshes
    :param transf:
    :return:
    '''
    for i in range(len(meshes)):
        meshes[i] = meshes[i].apply_transform(transf)