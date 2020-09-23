# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from OpenGL.GLUT import *
from OpenGL.GLU import *
from renderer.shaders.framework import *

from renderer.glRenderer import glRenderer

_glut_window = None

'''
#Usage:
    render.set_smpl_mesh(v)     #v for vertex locations in(6890,3)
    render.setBackgroundTexture(rawImg) #Optional BG texture
    render.setWindowSize(rawImg.shape[1], rawImg.shape[0])      #Optional: window size
    render.show_once()
'''

class denseposeRenderer(glRenderer):

    def __init__(self, width=1600, height=1200, name='GL Renderer',
                 program_files=['renderer/shaders/simple140.fs', 'renderer/shaders/simple140.vs'], color_size=1, ms_rate=1):
        glRenderer.__init__(self, width, height, name, program_files, color_size, ms_rate)

        self.densepose_info = self.loadDensepose_info()

        #Densepose Specific 
        self.dp_faces = self.densepose_info['All_Faces']-1 #0~7828
        self.dp_vertexIndices = self.densepose_info['All_vertices']-1    #(1,7829)       #Vertex orders used in denpose info. There are repeated vetices

        #DP color information
        dp_color_seg = self.densepose_info['All_FaceIndices']     #(13774,1)
        dp_color_seg = np.repeat(dp_color_seg,3,axis=1) /100.0#24.0   #(13774,3)
        self.dp_color_seg = np.repeat( dp_color_seg.flatten()[:,None],3,axis=1)        #(41332,3)

        dp_color_U = self.densepose_info['All_U_norm']     #(7289,1)
        dp_color_U =  np.repeat(dp_color_U,3,axis=1)    #(13774,3)
        self.dp_color_U = dp_color_U[self.dp_faces.reshape([-1])]  #(41332,3)

        dp_color_V = self.densepose_info['All_V_norm']     #(7829,3)
        dp_color_V =  np.repeat(dp_color_V,3,axis=1)    #(13774,3)
        self.dp_color_V = dp_color_V[self.dp_faces.reshape([-1])]  #(41332,3)

    #make sure you have: /yourpath/renderer/densepose_uv_data/UV_Processed.mat
    def loadDensepose_info(self, dp_data_path= 'extra_data/densepose_uv_data/UV_Processed.mat'):
        
        #Load densepose data
        import scipy.io as sio
        densepose_info = None
        densepose_info = sio.loadmat(dp_data_path)      #All_FaceIndices (13774), All_Faces(13774), All_U(7829), All_U_norm(7829), All_V(7829), All_V_norm (7829), All_vertices (7829)
        assert densepose_info is not None
        # All_FaceIndices - part labels for each face
        # All_Faces - vertex indices for each face
        # All_vertices - SMPL vertex IDs for all vertices (note that one SMPL vertex can be shared across parts and thus appear in faces with different part labels)
        # All_U - U coordinates for all vertices
        # All_V - V coordinates for all vertices
        # All_U_norm - normalization factor for U coordinates to map them to [0, 1] interval
        # All_V_norm - normalization factor for V coordinates to map them to [0, 1] interval
        # vertexColor = densepose_info['All_U_norm']*255
        # vertexColor = np.zeros((v.shape[1], 3))
        # vertexColor[:,0] = densepose_info['All_U_norm'][:v.shape[1]].flatten()       #(6890,3)
        # vertexColor[:,1] = densepose_info['All_V_norm'][:v.shape[1]].flatten()       #(6890,3)

        # # faces = smplWrapper.f
        # v =v[0]  #(6890,3)
        # dp_vertex = v[densepose_info['All_vertices']-1]  #(1,7829,3)        #Considering repeatation
        # faces =densepose_info['All_Faces']-1 #0~7828
        # # vertexColor = densepose_info['All_FaceIndices']     #(13774,1)
        # # vertexColor = np.repeat(vertexColor,3,axis=1) /24.0   #(13774,3)

        # # vertexColor = densepose_info['All_U_norm']     #(13774,1)
        # vertexColor = densepose_info['All_V_norm']     #(13774,1)
        # vertexColor =  np.repeat(vertexColor,3,axis=1) 

        # # vertexColor[vertexColor!=2]*=0
        # vertexColor[vertexColor==2]=24
        return densepose_info


    #vertice: (6890,3)
    #colormode: ['seg', 'u', 'v']
    def set_mesh(self, vertices, _):
        
        if vertices.dtype != np.dtype('float64'):
            vertices = vertices.astype(np.float64)      #Should be DOUBLE

        #Change the vertex and 
        dp_vertex = vertices[self.dp_vertexIndices][0]  #(7829,3)        #Considering repeatation

        # if colormode=='seg': #segment   
        #     self.color_data = self.dp_color_seg
        # elif colormode=='v':
        #     self.color_data[:,1] = self.dp_color_V
        # elif colormode=='u':
        #     self.color_data = self.dp_color_U       #(41322,3)
        # else:
        #     assert False

        self.color_data = self.dp_color_U       #(41322,3)       
        self.color_data[:,1] = self.dp_color_V[:,1]
        self.color_data[:,2] = self.dp_color_seg[:,2]

        self.vertex_data = dp_vertex[self.dp_faces.reshape([-1])]       #(41322,3)
        self.vertex_dim = self.vertex_data.shape[1]
        self.n_vertices = self.vertex_data.shape[0]

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.vertex_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.color_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)


    def add_mesh(self, vertices, _, color=None):
        """
        Concatenate the new mesh data to self.vertex_data  (as if a single giant mesh)

        Args:
            input_vertices (np.ndarray): (verNum, 3).
            input_faces (np.ndarray): (faceNum, 3).
        """

        if vertices.dtype != np.dtype('float64'):
            vertices = vertices.astype(np.float64)      #Should be DOUBLE
        
        dp_vertex = vertices[self.dp_vertexIndices][0]  #(7829,3)        #Considering repeatation

        color_data = self.dp_color_U       #(41322,3)       
        color_data[:,1] = self.dp_color_V[:,1]
        color_data[:,2] = self.dp_color_seg[:,2]

        if self.vertex_data is None:
            self.vertex_data = dp_vertex[self.dp_faces.reshape([-1])]       #(41322,3)
            self.color_data = color_data
        
        else:       #Add the data
            input_vertices = dp_vertex[self.dp_faces.reshape([-1])]       #(41322,3)
            self.vertex_data = np.concatenate( (self.vertex_data, input_vertices), axis=0)    #(6870,3)
            self.color_data = np.concatenate( (self.color_data, color_data), axis=0)    #(6870,3)

        self.vertex_dim = self.vertex_data.shape[1]
        self.n_vertices = self.vertex_data.shape[0]

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.vertex_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.color_data, GL_STATIC_DRAW)


        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.meshindex_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
