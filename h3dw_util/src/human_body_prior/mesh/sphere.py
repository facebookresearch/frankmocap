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
import trimesh
from human_body_prior.tools.omni_tools import colors

__all__ = ['Sphere', 'points_to_spheres']


class Sphere(object):
    def __init__(self, center, radius):
        if(center.flatten().shape != (3,)):
            raise Exception("Center should have size(1,3) instead of %s" % center.shape)
        self.center = center.flatten()
        self.radius = radius

    def __str__(self):
        return "%s:%s" % (self.center, self.radius)

    def to_mesh(self, color=colors['red']):
        v = np.array([[0.0000, -1.000, 0.0000], [0.7236, -0.447, 0.5257],
                      [-0.278, -0.447, 0.8506], [-0.894, -0.447, 0.0000],
                      [-0.278, -0.447, -0.850], [0.7236, -0.447, -0.525],
                      [0.2765, 0.4472, 0.8506], [-0.723, 0.4472, 0.5257],
                      [-0.720, 0.4472, -0.525], [0.2763, 0.4472, -0.850],
                      [0.8945, 0.4472, 0.0000], [0.0000, 1.0000, 0.0000],
                      [-0.165, -0.850, 0.4999], [0.4253, -0.850, 0.3090],
                      [0.2629, -0.525, 0.8090], [0.4253, -0.850, -0.309],
                      [0.8508, -0.525, 0.0000], [-0.525, -0.850, 0.0000],
                      [-0.688, -0.525, 0.4999], [-0.162, -0.850, -0.499],
                      [-0.688, -0.525, -0.499], [0.2628, -0.525, -0.809],
                      [0.9518, 0.0000, -0.309], [0.9510, 0.0000, 0.3090],
                      [0.5876, 0.0000, 0.8090], [0.0000, 0.0000, 1.0000],
                      [-0.588, 0.0000, 0.8090], [-0.951, 0.0000, 0.3090],
                      [-0.955, 0.0000, -0.309], [-0.587, 0.0000, -0.809],
                      [0.0000, 0.0000, -1.000], [0.5877, 0.0000, -0.809],
                      [0.6889, 0.5257, 0.4999], [-0.262, 0.5257, 0.8090],
                      [-0.854, 0.5257, 0.0000], [-0.262, 0.5257, -0.809],
                      [0.6889, 0.5257, -0.499], [0.5257, 0.8506, 0.0000],
                      [0.1626, 0.8506, 0.4999], [-0.425, 0.8506, 0.3090],
                      [-0.422, 0.8506, -0.309], [0.1624, 0.8506, -0.499]])

        f = np.array([[15, 3, 13], [13, 14, 15], [2, 15, 14], [13, 1, 14], [17, 2, 14], [14, 16, 17],
                      [6, 17, 16], [14, 1, 16], [19, 4, 18], [18, 13, 19], [3, 19, 13], [18, 1, 13],
                      [21, 5, 20], [20, 18, 21], [4, 21, 18], [20, 1, 18], [22, 6, 16], [16, 20, 22],
                      [5, 22, 20], [16, 1, 20], [24, 2, 17], [17, 23, 24], [11, 24, 23], [23, 17, 6],
                      [26, 3, 15], [15, 25, 26], [7, 26, 25], [25, 15, 2], [28, 4, 19], [19, 27, 28],
                      [8, 28, 27], [27, 19, 3], [30, 5, 21], [21, 29, 30], [9, 30, 29], [29, 21, 4],
                      [32, 6, 22], [22, 31, 32], [10, 32, 31], [31, 22, 5], [33, 7, 25], [25, 24, 33],
                      [11, 33, 24], [24, 25, 2], [34, 8, 27], [27, 26, 34], [7, 34, 26], [26, 27, 3],
                      [35, 9, 29], [29, 28, 35], [8, 35, 28], [28, 29, 4], [36, 10, 31], [31, 30, 36],
                      [9, 36, 30], [30, 31, 5], [37, 11, 23], [23, 32, 37], [10, 37, 32], [32, 23, 6],
                      [39, 7, 33], [33, 38, 39], [12, 39, 38], [38, 33, 11], [40, 8, 34], [34, 39, 40],
                      [12, 40, 39], [39, 34, 7], [41, 9, 35], [35, 40, 41], [12, 41, 40], [40, 35, 8],
                      [42, 10, 36], [36, 41, 42], [12, 42, 41], [41, 36, 9], [38, 11, 37], [37, 42, 38],
                      [12, 38, 42], [42, 37, 10]]) - 1

        # return Mesh(v=v * self.radius + self.center, f=f, vc=np.tile(color, (v.shape[0], 1)))
        return trimesh.Trimesh(vertices=v * self.radius + self.center, faces=f, vertex_colors=np.tile(color, (v.shape[0], 1)))

    def has_inside(self, point):
        return np.linalg.norm(point - self.center) <= self.radius

    def intersects(self, sphere):
        return np.linalg.norm(sphere.center - self.center) < (self.radius + sphere.radius)

    def intersection_vol(self, sphere):
        if not self.intersects(sphere):
            return 0
        d = np.linalg.norm(sphere.center - self.center)
        R, r = (self.radius, sphere.radius) if (self.radius > sphere.radius) else (sphere.radius, self.radius)
        if R >= (d + r):
            return (4 * np.pi * (r ** 3)) / 3

        # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        return (np.pi * (R + r - d) ** 2 * (d ** 2 + 2 * d * r - 3 * r * r + 2 * d * R + 6 * r * R - 3 * R * R)) / (12 * d)

def points_to_spheres(points, radius=0.01, vc = colors['red']):
    '''

    :param points: Nx3 numpy array
    :param radius:
    :param vc: either a 3-element normalized RGB vector or a list of them for each point
    :return:
    '''
    spheres = []
    for id in range(len(points)):
        spheres.append(Sphere( center= points[id].reshape(-1,3), radius=radius ).to_mesh( color = vc if len(vc)==3 and not isinstance(vc[0], list) else vc[id]))
    return spheres
