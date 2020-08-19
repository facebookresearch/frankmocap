"""
This code is used to load mosh data of Human3.6M provided by HMR
"""
import os
import sys
import shutil
import os.path as osp
import numpy as np
import pdb
import cv2
import multiprocessing as mp

from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints



def render_smpl(m):
    # Create OpenDR renderer
    rn = ColoredRenderer()
    # Assign attributes to renderer
    w, h = (640, 480)
    rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array(
        [0, 0, 2.]), f=np.array([w, w])/2., c=np.array([w, h])/2., k=np.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=m, f=m.f, bgcolor=np.zeros(3))
    # Construct point light source
    rn.vc = LambertianPointLight(
        f=m.f,
        v=rn.v,
        num_verts=len(m),
        light_pos=np.array([-1000, -1000, -2000]),
        vc=np.ones_like(m)*.9,
        light_color=np.array([1., 1., 1.]))
    image = rn.r * 255
    return image


def render_image_single(smpl, smpl_pose, smpl_shape):
    smpl.pose[:] = smpl_pose
    smpl.betas[:] = smpl_shape
    render_img = render_smpl(smpl)
    x, y, delta = 136, 11, 385
    render_img = render_img[y:y+delta, x:x+delta]
    render_img = cv2.resize(render_img, (224,224))
    return render_img


def render_image(smpl, smpl_pose, smpl_shape):
    # original image
    all_imgs = list()
    for i in range(1, 4):
        smpl_pose[0] = 0
        smpl_pose[1] = i*0.5*np.pi
        smpl_pose[2] = 0
        render_img = render_image_single(smpl, smpl_pose, smpl_shape)
        all_imgs.append(np.fliplr(np.flipud(render_img)))
    return np.concatenate(all_imgs, axis=1)


def render_smpl_to_image(img, vert, cam, renderer):
    f = 5.
    tz = f / cam[0]
    inputSize = 224
    cam_for_render = 0.5 * inputSize * np.array([f, 1, 1])
    cam_t = np.array([cam[1], cam[2], tz])
    # Undo pre-processing.
    input_img = img/255.0
    rend_img = renderer(vert + cam_t, cam_for_render, img=input_img)
    return rend_img