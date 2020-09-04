"""
Renders mesh using OpenDr for visualization.
"""
import sys
import numpy as np
import cv2
import pdb
from PIL import Image, ImageDraw
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from PIL import Image, ImageDraw
import subprocess as sp


def render(verts, faces, cam, inputSize, img=None, get_visible_faces=False):
    # inputSize = 4096
    f = 5
    tz = f / cam[0]
    cam_for_render = 0.5 * inputSize * np.array([f, 1, 1])
    cam_t = np.array([cam[1], cam[2], tz])

    verts = verts+cam_t
    cam = cam_for_render

    if img is None:
        img = np.ones((inputSize, inputSize, 3), dtype=np.float32)

    renderer = SMPLRenderer(img_size=inputSize)
    return_value = renderer(img=img, cam=cam, 
        verts=verts, faces=faces,
        get_visible_faces=get_visible_faces, color_id=2)
    return return_value


def project_joints(X, camera):
    X = X[None, :]
    camera = camera[None, :]
    camera = camera.reshape(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    res = camera[:, :, 0] * X_trans.reshape(X_trans.shape[0], -1)
    res = res.reshape(X_trans.shape[0], X_trans.shape[1], -1)
    return res.squeeze()


colors = {
    # colorbline/print/copy safe:
    # the format is BGR
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
    'light_pink': [.9, .7, .7],  # This is used to do no-3d
     'light_green': [166/255.0, 178/255.0, 30/255.0],
    # 'light_purple': [212/255.0, 112/255.0, 132/255.0],
    # 'light_purple': [255/255.0, 204/255.0, 229/255.0],
    # 'light_purple': [160/255.0, 160/255.0, 160/255.0],
    # 'light_purple': [160/255.0, 160/255.0, 160/255.0],
    # 'light_purple': [204/255.0, 204/255.0, 0/255.0],
}


def get_visible_faces(visibility_image):
    vis_img = visibility_image.astype(np.int64)
    vis_img[vis_img == 4294967295] = -1
    visible = np.nonzero(vis_img.ravel() >= 0)[0]
    vis_faces = np.unique(vis_img.ravel()[visible])
    return vis_faces


class SMPLRenderer(object):
    def __init__(self,
                 img_size=224,
                 flength=500.
    ):
        self.w = img_size
        self.h = img_size
        self.flength = flength

    def __call__(self,
                 verts,
                 faces,
                 cam=None,
                 img=None,
                 do_alpha=False,
                 far=None,
                 near=None,
                 color_id=0,
                 img_size=None,
                 get_visible_faces=False):
        """
        cam is 3D [f, px, py]
        """
        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h = img_size[0]
            w = img_size[1]
        else:
            h = self.h
            w = self.w

        if cam is None:
            cam = [self.flength, w / 2., h / 2.]

        use_cam = ProjectPoints(
            f=cam[0] * np.ones(2),
            rt=np.zeros(3),
            t=np.zeros(3),
            k=np.zeros(5),
            c=cam[1:3])

        if near is None:
            near = np.maximum(np.min(verts[:, 2]) - 25, 0.1)
        if far is None:
            far = np.maximum(np.max(verts[:, 2]) + 25, 25)

        imtmp, visible_faces = render_model(
            verts,
            faces,
            w,
            h,
            use_cam,
            do_alpha=do_alpha,
            img=img,
            far=far,
            near=near,
            color_id=color_id,
            get_visible_faces=True)
        image = (imtmp * 255).astype('uint8')

        if get_visible_faces:
            return image, visible_faces
        else:
            return image

    def rotated(self,
                verts,
                deg,
                cam=None,
                axis='y',
                img=None,
                do_alpha=True,
                far=None,
                near=None,
                color_id=0,
                img_size=None):
        import math
        if axis == 'y':
            around = cv2.Rodrigues(np.array([0, math.radians(deg), 0]))[0]
        elif axis == 'x':
            around = cv2.Rodrigues(np.array([math.radians(deg), 0, 0]))[0]
        else:
            around = cv2.Rodrigues(np.array([0, 0, math.radians(deg)]))[0]
        center = verts.mean(axis=0)
        new_v = np.dot((verts - center), around) + center

        return self.__call__(
            new_v,
            cam,
            img=img,
            do_alpha=do_alpha,
            far=far,
            near=near,
            img_size=img_size,
            color_id=color_id)


def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    return np.dot(points, ry)


def simple_renderer(rn,
                    verts,
                    faces,
                    yrot=np.radians(120),
                    color=colors['light_pink']):
    # Rendered model color
    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))
    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))
    
    visibility_image = rn.visibility_image
    visible_faces = get_visible_faces(visibility_image)

    return rn.r, visible_faces


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha.astype(
        imtmp.dtype)))
    return im_RGBA


def append_alpha(imtmp):
    alpha = np.ones_like(imtmp[:, :, 0]).astype(imtmp.dtype)
    if np.issubdtype(imtmp.dtype, np.uint8):
        alpha = alpha * 255
    b_channel, g_channel, r_channel = cv2.split(imtmp)
    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha))
    return im_RGBA


def render_model(verts,
                 faces,
                 w,
                 h,
                 cam,
                 near=0.5,
                 far=25,
                 img=None,
                 do_alpha=False,
                 color_id=None,
                 get_visible_faces=False):
    rn = _create_renderer(
        w=w, h=h, near=near, far=far, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c)
    

    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    if color_id is None:
        color = colors['light_blue']
    else:
        color_list = list(colors.values())
        color = color_list[color_id % len(color_list)]

    imtmp, visible_faces = simple_renderer(rn, verts, faces, color=color)

    # If white bg, make transparent.
    if img is None and do_alpha:
        imtmp = get_alpha(imtmp)
    elif img is not None and do_alpha:
        imtmp = append_alpha(imtmp)

    return imtmp, visible_faces