# Copyright (c) Facebook, Inc. and its affiliates.

# Part of code is modified from https://github.com/facebookresearch/pytorch3d

import cv2
import os
import sys
import torch
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVOrthographicCameras,
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    BlendParams,
    MeshRasterizer,  
    SoftPhongShader,
)

class Pytorch3dRenderer(object):

    def __init__(self, img_size, mesh_color):
        self.device = torch.device("cuda:0")
        # self.render_size = 1920

        self.img_size = img_size

        # mesh color
        mesh_color = np.array(mesh_color)[::-1]
        self.mesh_color = torch.from_numpy(
            mesh_color.copy()).view(1, 1, 3).float().to(self.device)

        # renderer for large objects, such as whole body.
        self.render_size_large = 700
        lights = PointLights(
            ambient_color = [[1.0, 1.0, 1.0],],
            diffuse_color = [[1.0, 1.0, 1.0],],
            device=self.device, location=[[1.0, 1.0, -30]])
        self.renderer_large = self.__get_renderer(self.render_size_large, lights)

        # renderer for small objects, such as whole body.
        self.render_size_medium = 400
        lights = PointLights(
            ambient_color = [[0.5, 0.5, 0.5],],
            diffuse_color = [[0.5, 0.5, 0.5],],
            device=self.device, location=[[1.0, 1.0, -30]])
        self.renderer_medium = self.__get_renderer(self.render_size_medium, lights)


        # renderer for small objects, such as whole body.
        self.render_size_small = 200
        lights = PointLights(
            ambient_color = [[0.5, 0.5, 0.5],],
            diffuse_color = [[0.5, 0.5, 0.5],],
            device=self.device, location=[[1.0, 1.0, -30]])
        self.renderer_small = self.__get_renderer(self.render_size_small, lights)



    def __get_renderer(self, render_size, lights):

        cameras = FoVOrthographicCameras(
            device = self.device,
            znear=0.1,
            zfar=10.0,
            max_y=1.0,
            min_y=-1.0,
            max_x=1.0,
            min_x=-1.0,
            scale_xyz=((1.0, 1.0, 1.0),),  # (1, 3)
        )

        raster_settings = RasterizationSettings(
            image_size = render_size,
            blur_radius = 0,
            faces_per_pixel = 1,
        )
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color = (0,0,0))

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )

        return renderer
    

    def render(self, verts, faces, bg_img):
        verts = verts.copy()
        faces = faces.copy()

        # bbox for verts
        x0 = int(np.min(verts[:, 0]))
        x1 = int(np.max(verts[:, 0]))
        y0 = int(np.min(verts[:, 1]))
        y1 = int(np.max(verts[:, 1]))
        width = x1 - x0
        height = y1 - y0

        bbox_size = max(height, width)
        if bbox_size <= self.render_size_small:
            # print("Using small size renderer")
            render_size = self.render_size_small
            renderer = self.renderer_small
        else:
            if bbox_size <= self.render_size_medium:
                # print("Using medium size renderer")
                render_size = self.render_size_medium
                renderer = self.renderer_medium
            else:
                # print("Using large size renderer")
                render_size = self.render_size_large
                renderer = self.renderer_large
        
        # padding the tight bbox
        margin = int(max(width, height) * 0.1)
        x0 = max(0, x0-margin)
        y0 = max(0, y0-margin)
        x1 = min(self.img_size, x1+margin)
        y1 = min(self.img_size, y1+margin)

        # move verts to be in the bbox
        verts[:, 0] -= x0
        verts[:, 1] -= y0

        # normalize verts to (-1, 1)
        bbox_size = max(y1-y0, x1-x0)
        half_size = bbox_size / 2
        verts[:, 0] = (verts[:, 0] - half_size) / half_size
        verts[:, 1] = (verts[:, 1] - half_size) / half_size

        # the coords of pytorch-3d is (1, 1) for upper-left and (-1, -1) for lower-right
        # so need to multiple minus for vertices
        verts[:, :2] *= -1

        # shift verts along the z-axis
        verts[:, 2] /= 112
        verts[:, 2] += 5

        verts_tensor = torch.from_numpy(verts).float().unsqueeze(0).cuda()
        faces_tensor = torch.from_numpy(faces.copy()).long().unsqueeze(0).cuda()

        # set color
        mesh_color = self.mesh_color.repeat(1, verts.shape[0], 1)
        textures = Textures(verts_rgb = mesh_color)

        # rendering
        mesh = Meshes(verts=verts_tensor, faces=faces_tensor, textures=textures)

        # blending rendered mesh with background image
        rend_img = renderer(mesh)
        rend_img = rend_img[0].cpu().numpy()


        scale_ratio = render_size / bbox_size
        img_size_new = int(self.img_size * scale_ratio)
        bg_img_new = cv2.resize(bg_img, (img_size_new, img_size_new))

        x0 = max(int(x0 * scale_ratio), 0)
        y0 = max(int(y0 * scale_ratio), 0)
        x1 = min(int(x1 * scale_ratio), img_size_new)
        y1 = min(int(y1 * scale_ratio), img_size_new)

        h0 = min(y1-y0, render_size)
        w0 = min(x1-x0, render_size)

        y1 = y0 + h0
        x1 = x0 + w0

        rend_img_new = np.zeros((img_size_new, img_size_new, 4))
        rend_img_new[y0:y1, x0:x1, :] = rend_img[:h0, :w0, :]
        rend_img = rend_img_new

        alpha = rend_img[:, :, 3:4]
        alpha[alpha>0] = 1.0
        

        rend_img = rend_img[:, :, :3] 
        maxColor = rend_img.max()
        rend_img *= 255 /maxColor #Make sure <1.0
        rend_img = rend_img[:, :, ::-1]

        res_img = alpha * rend_img + (1.0 - alpha) * bg_img_new

        res_img = cv2.resize(res_img, (self.img_size, self.img_size))

        return res_img