# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from OpenGL.GLUT import *
from OpenGL.GLU import *
from renderer.shaders.framework import *

from renderer.glRenderer import glRenderer

# from renderer.render_utils import ComputeNormal

_glut_window = None

'''
#Usage:
    render.set_smpl_mesh(v)     #v for vertex locations in(6890,3)
    render.setBackgroundTexture(rawImg) #Optional BG texture
    render.setWindowSize(rawImg.shape[1], rawImg.shape[0])      #Optional: window size
    render.show_once()
'''

class meshRenderer(glRenderer):

    def __init__(self, width=1600, height=1200, name='GL Renderer',
                #  program_files=['renderer/shaders/simple140.fs', 'renderer/shaders/simple140.vs'],
                #  program_files=['renderer/shaders/normal140.fs', 'renderer/shaders/normal140.vs'],
                # program_files=['renderer/shaders/geo140.fs', 'renderer/shaders/geo140.vs'],
                render_mode ="normal",  #color, geo, normal
                color_size=1, ms_rate=1):

        self.render_mode = render_mode
        self.program_files ={}
        self.program_files['color'] = ['renderer/shaders/simple140.fs', 'renderer/shaders/simple140.vs']
        self.program_files['normal'] = ['renderer/shaders/normal140.fs', 'renderer/shaders/normal140.vs']
        self.program_files['geo'] = ['renderer/shaders/colorgeo140.fs', 'renderer/shaders/colorgeo140.vs']

        glRenderer.__init__(self, width, height, name, self.program_files[render_mode], color_size, ms_rate)

    def setRenderMode(self, render_mode):
        """
        Set render mode among ['color', 'normal', 'geo']
        """
        if self.render_mode == render_mode:
            return
        
        self.render_mode = render_mode
        self.initShaderProgram(self.program_files[render_mode])


    def drawMesh(self):

        if self.vertex_dim is None:
            return
        # self.draw_init()

        glColor3f(1,1,0)
        glUseProgram(self.program)
        
        mvMat = glGetFloatv(GL_MODELVIEW_MATRIX)
        pMat = glGetFloatv(GL_PROJECTION_MATRIX)
        # mvpMat = pMat*mvMat

        self.model_view_matrix = mvMat
        self.projection_matrix = pMat

        # glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE, self.model_view_matrix.transpose())
        # glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE, self.projection_matrix.transpose())
        glUniformMatrix4fv(self.model_mat_unif, 1, GL_FALSE, self.model_view_matrix)
        glUniformMatrix4fv(self.persp_mat_unif, 1, GL_FALSE, self.projection_matrix)

        # Handle vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, self.vertex_dim, GL_DOUBLE, GL_FALSE, 0, None)

        # # Handle normal buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, None)

        # # Handle color buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_DOUBLE, GL_FALSE, 0, None)
        

        if True:#self.meshindex_data:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)           #Note "GL_ELEMENT_ARRAY_BUFFER" instead of GL_ARRAY_BUFFER
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.meshindex_data, GL_STATIC_DRAW)

        # glDrawArrays(GL_TRIANGLES, 0, self.n_vertices)
        glDrawElements(GL_TRIANGLES, len(self.meshindex_data), GL_UNSIGNED_INT, None)       #For index array (mesh face data)
        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glUseProgram(0)