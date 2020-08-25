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
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import makepath
import os
import trimesh

def smpl_params2ply(bm, out_dir, pose_body, pose_hand = None, trans=None, betas=None, root_orient=None):
    '''
    :param bm: pytorch body model with batch_size 1
    :param pose_body: can be a single list of pose parameters, or a list of list of pose parameters:
    :param trans: Nx3
    :param betas: Nxnum_betas
    :return
    dumps are all parameter as gltf objects
    '''

    faces = c2c(bm.f)

    makepath(out_dir)

    for fIdx in range(0, len(pose_body)):

        bm.pose_body.data[0,:] = bm.pose_body.new(pose_body[fIdx].reshape(1,-1))
        if pose_hand is not None: bm.pose_hand.data[0,:] = bm.pose_hand.new(pose_hand[fIdx].reshape(1,-1))
        if trans is not None: bm.trans.data[0,:] = bm.trans.new(trans[fIdx].reshape(1,-1))
        if betas is not None: bm.betas.data[0,:len(betas[fIdx])] = bm.betas.new(betas[fIdx])
        if root_orient is not None: bm.root_orient.data[0,:] = bm.root_orient.new(root_orient[fIdx])

        v = c2c(bm.forward().v)[0]

        mesh = trimesh.base.Trimesh(v, faces)
        mesh.export(os.path.join(out_dir, '%03d.ply' % fIdx))


def render_smpl_params(bm, pose_body, pose_hand = None, trans=None, betas=None, root_orient=None):
    '''
    :param bm: pytorch body model with batch_size 1
    :param pose_body: Nx21x3
    :param trans: Nx3
    :param betas: Nxnum_betas
    :return: N x 400 x 400 x 3
    '''

    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    from human_body_prior.tools.omni_tools import colors
    from human_body_prior.mesh.mesh_viewer import MeshViewer
    faces = c2c(bm.f)

    imw, imh = 400, 400

    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    images = []
    for fIdx in range(0, len(pose_body)):

        bm.pose_body.data[0,:] = bm.pose_body.new(pose_body[fIdx].reshape(1,-1))
        if pose_hand is not None: bm.pose_hand.data[0,:] = bm.pose_hand.new(pose_hand[fIdx])
        if trans is not None: bm.trans.data[0,:] = bm.trans.new(trans[fIdx])
        if betas is not None: bm.betas.data[0,:len(betas[fIdx])] = bm.betas.new(betas[fIdx])
        if root_orient is not None: bm.root_orient.data[0,:] = bm.root_orient.new(root_orient[fIdx])

        v = c2c(bm.forward().v)[0]

        mesh = trimesh.base.Trimesh(v, faces, vertex_colors=np.ones_like(v)*colors['grey'])
        mv.set_meshes([mesh], 'static')

        images.append(mv.render())

    return np.array(images).reshape(len(pose_body), imw, imh, 3)

def imagearray2file(img_array, outpath=None, fps=30):
    '''
    :param nparray: RxCxTxwidthxheightx3
    :param outpath: the directory where T images will be dumped for each time point in range T
    :param fps: fps of the gif file
    :return:
        it will return an image list with length T
        if outpath is given as a png file, an image will be saved for each t in T.
        if outpath is given as a gif file, an animated image with T frames will be created.
    '''
    import cv2
    from human_body_prior.tools.omni_tools import makepath

    if outpath is not None:
        makepath(outpath, isfile=True)

    if not isinstance(img_array, np.ndarray) or img_array.ndim < 6:
        raise ValueError('img_array should be a numpy array of shape RxCxTxwidthxheightx3')

    R, C, T, img_h, img_w, img_c = img_array.shape

    out_images = []
    for tIdx in range(T):
        row_images = []
        for rIdx in range(R):
            col_images = []
            for cIdx in range(C):
                col_images.append(img_array[rIdx, cIdx, tIdx])
            row_images.append(np.hstack(col_images))
        t_image = np.vstack(row_images)
        out_images.append(t_image)

    if outpath is not None:
        if '.png' in outpath:
            for tIdx in range(T):
                if T > 1:
                    cur_outpath = outpath.replace('.png', '_%03d.png'%tIdx)
                else:
                    cur_outpath = outpath
                cv2.imwrite(cur_outpath, out_images[tIdx])
                while not os.path.exists(cur_outpath): continue  # wait until the snapshot is written to the disk
        elif '.gif' in outpath:
            import imageio
            with imageio.get_writer(outpath, mode='I', fps = fps) as writer:
                for tIdx in range(T):
                    img = out_images[tIdx].astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    writer.append_data(img)
        elif '.avi' in outpath:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(outpath, fourcc, fps, (img_w, img_h), True)
            for tIdx in range(T):
                img = out_images[tIdx].astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)

            video.release()
            cv2.destroyAllWindows()
        elif '.mp4' in outpath:

            from moviepy.editor import ImageSequenceClip
            animation = ImageSequenceClip(out_images, fps=fps)
            animation.write_videofile(outpath)

    return out_images