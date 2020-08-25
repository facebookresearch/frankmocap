import cv2
import keyboard
import numpy as np
import open3d as o3d
import pygame
from transforms3d.axangles import axangle2mat

import config
# from capture import OpenCVCapture
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from utils import OneEuroFilter, imresize
from wrappers import ModelPipeline
from utils import *
import os, sys, shutil
import os.path as osp
import pdb
import ry_utils

def main():
    """
    Launch an application that reads from a webcam and estimates hand pose at
    real-time.

    The captured hand must be the right hand, but will be flipped internally
    and rendered.

    Parameters
    ----------
    capture : object
      An object from `capture.py` to read capture stream from.
    """
    ############ output visualization ############
    view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
    window_size = 256

    hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = \
      o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
    mesh.compute_vertex_normals()

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
      width=window_size + 1, height=window_size + 1,
      window_name='Minimal Hand - output'
    )
    viewer.add_geometry(mesh)

    view_control = viewer.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = cam_params.extrinsic.copy()
    extrinsic[0:3, 3] = 0
    cam_params.extrinsic = extrinsic
    cam_params.intrinsic.set_intrinsics(
      window_size + 1, window_size + 1, config.CAM_FX, config.CAM_FY,
      window_size // 2, window_size // 2
    )
    view_control.convert_from_pinhole_camera_parameters(cam_params)
    view_control.set_constant_z_far(1000)

    render_option = viewer.get_render_option()
    render_option.load_from_json('./render_option.json')
    viewer.update_renderer()

    ############ misc ############
    mesh_smoother = OneEuroFilter(4.0, 0.0)
    model = ModelPipeline()

    seq_name = "body_only"
    in_dir = osp.join("/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/body_capture/image_hand", seq_name)
    out_dir = osp.join("render_hand", seq_name)
    ry_utils.renew_dir(out_dir)

    img_list = sorted([file for file in os.listdir(in_dir) if file.endswith('.png')])

    for file in img_list:
        img_path = osp.join(in_dir, file)
        frame_large = cv2.imread(img_path)[:, :, ::-1]
        if frame_large.shape[0] > frame_large.shape[1]:
          margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
          # frame_large = frame_large[margin:-margin]
          frame_large = frame_large[margin:frame_large.shape[0]-margin, :, :]
        else:
          margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
          frame_large = frame_large[:, margin:frame_large.shape[1]-margin, :]

        # frame_large = np.flip(frame_large, axis=1).copy()
        frame_large = np.fliplr(frame_large).copy()
        frame = imresize(frame_large, (128, 128))

        _, theta_mpii = model.process(frame)
        theta_mano = mpii_to_mano(theta_mpii)

        v = hand_mesh.set_abs_quat(theta_mano)
        v *= 2 # for better visualization
        v = v * 1000 + np.array([0, 0, 400])
        v = mesh_smoother.process(v)
        mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
        mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
        mesh.paint_uniform_color(config.HAND_COLOR)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        viewer.update_geometry()

        viewer.poll_events()
        viewer.capture_screen_image("0.png", False)

        img = cv2.imread("0.png")
        img = np.fliplr(img)
        frame_large = cv2.resize(np.fliplr(frame_large), img.shape[:2])[:, :, ::-1]
        res_img = np.concatenate((frame_large, img), axis=1)
        cv2.imwrite(osp.join(out_dir, file), res_img)

    '''
    display.blit(
      pygame.surfarray.make_surface(
        np.transpose(
          imresize(frame_large, (window_size, window_size)
        ), (1, 0, 2))
      ),
      (0, 0)
    )
    pygame.display.update()

    if keyboard.is_pressed("esc"):
      break

    clock.tick(30)
    '''


if __name__ == '__main__':
    main()