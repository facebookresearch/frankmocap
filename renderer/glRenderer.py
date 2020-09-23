# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from renderer.shaders.framework import createProgram, loadShader

from renderer.render_utils import ComputeNormal

import cv2

_glut_window = None

class glRenderer:

    def __init__(self, width=640, height=480, name='GL Renderer',
                 program_files=['renderer/shaders/simple140.fs', 'renderer/shaders/simple140.vs'], color_size=1, ms_rate=1):
        self.width = width
        self.height = height
        self.name = name
        self.display_mode = GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE
        self.use_inverse_depth = False

        global _glut_window
        if _glut_window is None:
            glutInit()
            glutInitDisplayMode(self.display_mode)
            glutInitWindowSize(self.width, self.height)
            glutInitWindowPosition(0, 0)
            _glut_window = glutCreateWindow("GL_Renderer")


            glEnable(GL_DEPTH_CLAMP)
            glEnable(GL_DEPTH_TEST)

            glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE)
            glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE)
            glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE)


        self.program = None
        self.initShaderProgram(program_files)

        # Init Uniform variables
        self.model_mat_unif = glGetUniformLocation(self.program, 'ModelMat')
        self.persp_mat_unif = glGetUniformLocation(self.program, 'PerspMat')

        self.model_view_matrix = None
        self.projection_matrix = None

        self.vertex_buffer = glGenBuffers(1)
        self.color_buffer = glGenBuffers(1)
        self.normal_buffer = glGenBuffers(1)

        self.index_buffer = glGenBuffers(1)     #for Mesh face indices. Without this vertices should be repeated and ordered (3x times bigger)

        #Create Background Texture
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)       #So that texture doesnt have to be power of 2
        self.backgroundTextureID = glGenTextures(1)
        K = np.array([[2000, 0, 960],[0, 2000, 540],[0,0,1]])       #MTC default camera. for 1920 x 1080 input image
        self.camView_K = K
        
        # Inner storage for buffer data
        self.vertex_data = None
        self.vertex_dim = None
        self.n_vertices = None

        #Variables for view change
        self.m_xTrans = 0.
        self.m_yTrans = 0.
        self.m_zTrans = 0.
        self.m_zoom = 378
        self.m_xRotate = 59.
        self.m_yRotate = 0.
        self.m_zRotate = 0.
        self.m_xrot = 0.0
        self.m_yrot = 0.0

        #Camera view
        self.m_viewmode = "cam"

        #To compute sideview
        self.m_meshCenter = None

        #Option
        self.bOffscreenMode = False
        self.bAntiAliasing = True #apply anti aliasing

        #Textures
        self.data_texture = None


        #Visualization option
        self.bShowBackground = False
        self.bShowFloor = False

        self.nearPlane = 1  # original
        self.farPlane = 10000.0

        self.counter=1


        self.bOrthoCam = True
        
        glutMouseFunc(self.mouseButton)
        glutMotionFunc(self.mouseMotion)
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
    
    def initShaderProgram(self, program_files):
         # Init shader programs
        shader_list = []
        for program_file in program_files:
            _, ext = os.path.splitext(program_file)
            if ext == '.vs':
                shader_list.append(loadShader(GL_VERTEX_SHADER, program_file))
            elif ext == '.fs':
                shader_list.append(loadShader(GL_FRAGMENT_SHADER, program_file))
            elif ext == '.gs':
                shader_list.append(loadShader(GL_GEOMETRY_SHADER, program_file))

        if self.program is not None:
            glDeleteProgram(self.program)

        self.program = createProgram(shader_list)
        for shader in shader_list:
            glDeleteShader(shader)

    def reshape(self,width, height):

        if self.bOffscreenMode ==False:     #Offscreen mode doesn't allow reshape
            self.width = width
            self.height = height
            glViewport(0, 0, self.width, self.height)
            print("reshape: {}, {}".format(self.width, self.height))

        # glViewport(0, 0, 500,500)

    def setWindowSize(self,new_width, new_height):

        if new_width != self.width or  new_height!=self.height:
            self.width = new_width
            self.height =new_height
            glutReshapeWindow(self.width,self.height)

        #Neet to refresh opengl to apply resizing (the number of required iterations is a bit random)
        # glutPostRedisplay()
        glutMainLoopEvent()
        iterNum =0
        while True:
            curWidth =glutGet(GLUT_WINDOW_WIDTH)
            curHeight =  glutGet(GLUT_WINDOW_HEIGHT)
            # print("curWindSize: {}, {}".format(curWidth, curHeight))

            if curWidth == new_width and curHeight == new_height:
                break
            glutPostRedisplay()
            glutMainLoopEvent()
            iterNum+=1  
            if iterNum>20:
                print("Wraning: Cannot resize the gl window")
                break

        # print("{} refreshing is done to resize window".format(iterNum))

       
    def setViewportSize(self,new_width, new_height):
        if new_width != self.width or  new_height!=self.height:
            self.width = new_width
            self.height =new_height
        glViewport(0, 0, self.width, self.height)
        # print("{}, {}".format(self.width, self.height))

    def setCameraViewOrth(self):

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # texHeight,texWidth =  self.data_texture.shape[:2]
        # # texHeight,texWidth =   1024, 1024
        # texHeight*=0.5
        # texWidth*=0.5
        texHeight,texWidth =  self.height, self.width
        # texHeight,texWidth =   1024, 1024
        texHeight*=0.5
        texWidth*=0.5
        # texHeight *=BACKGROUND_IMAGE_PLANE_DEPTH
        # texWidth *=BACKGROUND_IMAGE_PLANE_DEPTH

        glOrtho(-texWidth, texWidth, -texHeight, texHeight, -1500, 1500)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)
        # glMultMatrixd(camMatrix.T)
    

    # Show the world in a camera cooridnate (defined by K)
    def setCameraView(self):
     
        invR = np.eye(3)
        invT = np.zeros((3, 1))
        # invT[2] = 400
        camMatrix = np.hstack((invR, invT))
        # denotes camera matrix, [R|t]
        camMatrix = np.vstack((camMatrix, [0, 0, 0, 1]))

        if self.camView_K is None:
            # print(
            #     "## Warning: no K is set, so I use a default cam parameter defined for MTC"
            # )
            # setCamView_K_DefaultForMTC()
            assert False
        K = self.camView_K.copy()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Kscale = 1920.0/g_Width
        Kscale = 1.0  # 1920.0/g_Width        :: why do we need this?
        K = K / Kscale
        ProjM = np.zeros((4, 4))
        ProjM[0, 0] = 2 * K[0, 0] / self.width
        ProjM[0, 2] = (self.width - 2 * K[0, 2]) / self.width
        ProjM[1, 1] = 2 * K[1, 1] / self.height
        ProjM[1, 2] = (-self.height + 2 * K[1, 2]) / self.height

        ProjM[2, 2] = (-self.farPlane - self.nearPlane) / (self.farPlane - self.nearPlane)
        ProjM[2, 3] = -2 * self.farPlane * self.nearPlane / (self.farPlane - self.nearPlane)
        ProjM[3, 2] = -1

        glLoadMatrixd(ProjM.T)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)
        glMultMatrixd(camMatrix.T)

    def setWorldCenterBySceneCenter(self):
        self.m_meshCenter  =np.mean(self.vertex_data,axis=0)

        meshMin = np.min(self.vertex_data,axis=0)
        meshMax = np.max(self.vertex_data,axis=0)

        self.meshWidth = max(meshMax-meshMin)



        self.m_xTrans = -self.m_meshCenter[0]# +self.counter
        self.m_yTrans = -self.m_meshCenter[1]
        self.m_zTrans = -self.m_meshCenter[2]

        self.m_zoom = min(self.m_zoom, 1000)
        self.m_zoom = 120 * max(self.meshWidth,100)/100

    def setZoom(self, z):
        self.m_zoom = z

    def setSideView(self):
        
        if self.m_meshCenter is None:
            self.setWorldCenterBySceneCenter()

        # self.m_zoom =400

        self.m_yRotate = 0 
        self.m_xRotate = 90 #+ self.counter


        glMatrixMode(GL_MODELVIEW)

        #Zoom out first
        glTranslatef(0,0,self.m_zoom)
        print(f"Zoom: {self.m_zoom}")
        
        #Rotate (to this end, around the center point)
        glRotatef( -self.m_yRotate, 1.0, 0.0, 0.0)
        glRotatef( -self.m_xRotate, 0.0, 1.0, 0.0)
        # glRotatef( self.m_zRotate, 0.0, 0.0, 1.0)

        # print(f"{self.m_xTrans}, {self.m_yTrans}, {self.m_zTrans}, {self.m_zoom}")
        # glTranslatef(0,0,self.m_zoom)
        glTranslatef( self.m_xTrans,  self.m_yTrans, self.m_zTrans)


    def setViewAngle(self,azimuthAng, elevationAng):
        """
        azimuthAng: 90 for sideview
        elevationAng: 90 for topview
        """
        self.m_xRotate = azimuthAng
        self.m_yRotate = -elevationAng

    def setFree3DView(self):
        if self.m_meshCenter is None:
            self.setWorldCenterBySceneCenter()

        glMatrixMode(GL_MODELVIEW)
        # glLoadIdentity()
        # gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)

        #Zoom out first
        glTranslatef(0,0,self.m_zoom)

        #Rotate (to this end, around the center point)
        glRotatef( -self.m_yRotate, 1.0, 0.0, 0.0)
        glRotatef( -self.m_xRotate, 0.0, 1.0, 0.0)
        # glRotatef( self.m_zRotate, 0.0, 0.0, 1.0)

        # print(f"{self.m_xTrans}, {self.m_yTrans}, {self.m_zTrans}, {self.m_zoom}")
        # glTranslatef(0,0,self.m_zoom)
        glTranslatef( self.m_xTrans,  self.m_yTrans, self.m_zTrans)
        

    def showBackground(self, bShow):
        self.bShowBackground = bShow

    def setBackgroundTexture(self,img):
        self.data_texture = img
    
    
    # 3x3 intrinsic camera matrix
    def setCamView_K(self, K):
        self.camView_K = K

    def setOrthoCamera(self, bOrtho = True):
        self.bOrthoCam = bOrtho

    def drawBackgroundOrth(self):

        if self.data_texture is None:
            return

        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)

        # glUseProgram(0)

        glBindTexture(GL_TEXTURE_2D, self.backgroundTextureID)
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, self.data_texture)
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 480, 640, 0, GL_RGB, GL_UNSIGNED_BYTE, self.data_texture.data)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.data_texture.shape[1], self.data_texture.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, self.data_texture.data)
        texHeight,texWidth =   self.data_texture.shape[:2]
        texHeight*=0.5
        texWidth*=0.5

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBegin(GL_QUADS)
        glColor3f(1.0, 1.0, 1.0)
        # d = BACKGROUND_IMAGE_PLANE_DEPTH
        d = 10

        glTexCoord2f(0, 0)
        # #Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K(m_options.m_pK);
        P = np.array([-texWidth, -texHeight, d])
        # P = np.matmul(K_inv,P)
        # P = P / P[2]
        glVertex3f(P[0] , P[1] , P[2] );  # K^{-1} [0, 0, 1]^T

        glTexCoord2f(1, 0)
        # P = [1920, 0, 1]
        P = [texWidth, -texHeight, d]
        glVertex3f(P[0] , P[1] , P[2] );  # K^{-1} [0, 0, 1]^T

        glTexCoord2f(1, 1)
        # P = [1920, 1080, 1]
        P = [texWidth, texHeight, d]
        # P = np.matmul(K_inv,P)
        # P = P / P[2]
        # glVertex3f(P[0] * d, P[1] * d, P[2] * d)
        glVertex3f(P[0] , P[1] , P[2] );  # K^{-1} [0, 0, 1]^T

        glTexCoord2f(0, 1)
        # P = [0, 1080, 1]
        P = [-texWidth, texHeight, d]
        # glVertex3f(P[0] * d, P[1] * d, P[2] * d)
        glVertex3f(P[0] , P[1] , P[2] );  # K^{-1} [0, 0, 1]^T
        glEnd()

        glEnable(GL_LIGHTING)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        
    def drawBackgroundPersp(self):

        if self.data_texture is None:
            return

        # if self.camView_K is None:
        #     self.setCamView_K_DefaultForMTC()

        K_inv = np.linalg.inv(self.camView_K)

        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)

        # glUseProgram(0)

        glBindTexture(GL_TEXTURE_2D, self.backgroundTextureID)
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1920, 1080, 0, GL_RGB, GL_UNSIGNED_BYTE, self.data_texture)
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 480, 640, 0, GL_RGB, GL_UNSIGNED_BYTE, self.data_texture.data)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.data_texture.shape[1], self.data_texture.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, self.data_texture.data)
        texHeight,texWidth =   self.data_texture.shape[:2]

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)


        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBegin(GL_QUADS)
        glColor3f(1.0, 1.0, 1.0)
        BACKGROUND_IMAGE_PLANE_DEPTH=3000
        d = BACKGROUND_IMAGE_PLANE_DEPTH

        glTexCoord2f(0, 0)
        # #Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K(m_options.m_pK);
        P = np.array([0, 0, 1])
        P = np.matmul(K_inv,P)
        P = P / P[2]
        glVertex3f(P[0] * d, P[1] * d, P[2] * d);  # K^{-1} [0, 0, 1]^T

        glTexCoord2f(1, 0)
        # P = [1920, 0, 1]
        P = [texWidth, 0, 1]

        P = np.matmul(K_inv,P)
        P = P / P[2]
        glVertex3f(P[0] * d, P[1] * d, P[2] * d);  # K^{-1} [0, 0, 1]^T

        glTexCoord2f(1, 1)
        # P = [1920, 1080, 1]
        P = [texWidth, texHeight, 1]
        P = np.matmul(K_inv,P)
        P = P / P[2]
        glVertex3f(P[0] * d, P[1] * d, P[2] * d)

        glTexCoord2f(0, 1)
        # P = [0, 1080, 1]
        P = [0, texHeight, 1]
        P = np.matmul(K_inv,P)
        P = P / P[2]
        glVertex3f(P[0] * d, P[1] * d, P[2] * d)
        glEnd()

        glEnable(GL_LIGHTING)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)


    def mouseButton(self, button, state, x, y):
        # global self.m_action, self.m_xMousePtStart, self.m_yMousePtStart
        if (button==GLUT_LEFT_BUTTON):
            if (glutGetModifiers() == GLUT_ACTIVE_SHIFT):
                self.m_action = "TRANS"
            else:
                self.m_action = "MOVE_EYE"
        #elif (button==GLUT_MIDDLE_BUTTON):
        #    action = "TRANS"
        elif (button==GLUT_RIGHT_BUTTON):
            self.m_action = "ZOOM"
        self.m_xMousePtStart = x
        self.m_yMousePtStart = y

    def mouseMotion(self, x, y):
        # global self.m_zoom, self.m_xMousePtStart, self.m_yMousePtStart, self.m_xRotate, self.m_yRotate, self.m_zRotate, self.m_xTrans, self.m_zTrans
        if (self.m_action=="MOVE_EYE"):
            self.m_xRotate += x - self.m_xMousePtStart
            self.m_yRotate -= y - self.m_yMousePtStart
        elif (self.m_action=="MOVE_EYE_2"):
            self.m_zRotate += y - self.m_yMousePtStart
        elif (self.m_action=="TRANS"):
            self.m_xTrans += x - self.m_xMousePtStart
            self.m_zTrans += y - self.m_yMousePtStart
        elif (self.m_action=="ZOOM"):
            self.m_zoom -= y - self.m_yMousePtStart
            # print(self.m_zoom)
        else:
            print("unknown action\n", self.m_action)
        self.m_xMousePtStart = x
        self.m_yMousePtStart = y

        # print ('xTrans {},  yTrans {}, zoom {} xRotate{} yRotate {} zRotate {}'.format(self.m_xTrans,  self.m_yTrans,  self.m_zoom,  self.m_xRotate,  self.m_yRotate,  self.m_zRotate))
        glutPostRedisplay()

    def set_viewpoint(self, projection, model_view):
        self.projection_matrix = projection
        self.model_view_matrix = model_view

    def get_screen_color_fbgra(self):
        glReadBuffer(GL_BACK)   #GL_BACK is Default in double buffering
        data = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_FLOAT, outputType=None)
        rgb = data.reshape(self.height, self.width, -1)
        rgb = np.flip(rgb, 0)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGRA)
        return rgb
    
    def get_screen_color_ibgr(self):
        glReadBuffer(GL_BACK)   #GL_BACK is Default in double buffering
        data = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_FLOAT, outputType=None)
        rgb = data.reshape(self.height, self.width, -1)
        rgb = np.flip(rgb, 0)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        rgb = (rgb*255).astype(np.uint8)
        return rgb
    
    def setCameraViewMode(self, viewmode="cam"):
        """
        viewmode
            cam: cam view
            side: side view
            free: free view point
        """
        self.m_viewmode = viewmode

    def get_z_value(self):
        glReadBuffer(GL_BACK)   #GL_BACK is Default in double buffering

        data = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT, outputType=None)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        z = data.reshape(self.height, self.width)
        z = np.flip(z, 0)
        return z
    
    def drawFloor(self):
        glDisable(GL_LIGHTING)

        glPolygonMode(GL_FRONT, GL_FILL)
        glPolygonMode(GL_BACK, GL_FILL)
        # glPolygonMode(GL_FRONT, GL_FILL)
        gridNum = 10
        width = 200
        halfWidth =width/2
        # g_floorCenter = np.array([0,0.5,0])

        g_floorCenter = np.array([100,100,0])
        g_floorAxis1 = np.array([1,0,0])
        g_floorAxis2 = np.array([0,0,1])

        origin = g_floorCenter - g_floorAxis1*(width*gridNum/2 ) - g_floorAxis2*(width*gridNum/2)
        axis1 =  g_floorAxis1 * width
        axis2 =  g_floorAxis2 * width
        for y in range(gridNum+1):
            for x in range(gridNum+1):

                if (x+y) % 2 ==0:
                    glColor(1.0,1.0,1.0,1.0) #white
                else:
                    # glColor(0.95,0.95,0.95,0.3) #grey
                    glColor(0.3,0.3,0.3,0.5) #grey

                p1 = origin + axis1*x + axis2*y
                p2 = p1+ axis1
                p3 = p1+ axis2
                p4 = p1+ axis1 + axis2

                glBegin(GL_QUADS)
                glVertex3f(   p1[0], p1[1], p1[2])
                glVertex3f(   p2[0], p2[1], p2[2])
                glVertex3f(   p4[0], p4[1], p4[2])
                glVertex3f(   p3[0], p3[1], p3[2])
                glEnd()

        glEnable(GL_LIGHTING)
        
                
    def set_mesh(self, input_vertices, input_faces, color=None):
          
        #Compute normal
        if True:#bComputeNormal:
        # print("## setMeshData: Computing face normals automatically.")
            vertices_temp = input_vertices[np.newaxis,:,:]
            input_normal = ComputeNormal(vertices_temp,input_faces) #output: (N, 18540, 3)

        if input_vertices.dtype != np.dtype('float64'):
            input_vertices = input_vertices.astype(np.float64)      #Should be DOUBLE

        #Change the vertex and 
        # dp_vertex = vertices[self.dp_vertexIndices][0]  #(7829,3)        #Considering repeatation

        # if colormode=='normal': #segment   
        #     self.color_data = self.dp_color_seg
        # else:
        #     assert False
        
        input_faces_ = input_faces.flatten()

        if False:#Without index buffer
            self.vertex_data = input_vertices[input_faces_,:]   
            self.normal_data = input_normal[0][input_faces_,:]  

        else: #With index buffer
            self.vertex_data = input_vertices   #(6870,3)
            self.normal_data = input_normal[0]  #(18540,3)

            if color is None:
                self.color_data = np.ones(input_vertices.shape)   #(6870,3)
            else:
                self.color_data = np.tile(color, (input_vertices.shape[0], 1) )   #(6870,3)

            self.meshindex_data = input_faces_
            self.meshindex_data = self.meshindex_data.astype(np.int32)      #Should be DOUBLE
            

        self.vertex_dim = self.vertex_data.shape[1]
        self.n_vertices = self.vertex_data.shape[0]

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.vertex_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.normal_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.color_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.meshindex_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def clear_mesh(self):
        """
        Clear up all mesh 
        """
        self.vertex_data = None
        self.normal_data = None
        self.meshindex_data = None

    def add_mesh(self, input_vertices, input_faces, color=None):
        """
        Concatenate the new mesh data to self.vertex_data  (as if a single giant mesh)

        Args:
            input_vertices (np.ndarray): (verNum, 3).

            input_faces (np.ndarray): (faceNum, 3).
        """

          
        #Compute normal
        if True:#bComputeNormal:
            vertices_temp = input_vertices[np.newaxis,:,:]
            input_normal = ComputeNormal(vertices_temp,input_faces) #output: (N, 18540, 3)

        if input_vertices.dtype != np.dtype('float64'):
            input_vertices = input_vertices.astype(np.float64)      #Should be DOUBLE

        #Change the vertex and 
        # dp_vertex = vertices[self.dp_vertexIndices][0]  #(7829,3)        #Considering repeatation
        # if colormode=='normal': #segment   
        #     self.color_data = self.dp_color_seg
        # else:
        #     assert False
        
        input_faces_ = input_faces.flatten()

        if self.vertex_data is None:
            self.vertex_data = input_vertices   #(6870,3)
            self.normal_data = input_normal[0]  #(6870,3)

            if color is None:
                self.color_data = np.ones(input_vertices.shape)   #(6870,3)
            else:
                self.color_data = np.tile(color, (input_vertices.shape[0], 1) )   #(6870,3)

            self.meshindex_data = input_faces_
            self.meshindex_data = self.meshindex_data.astype(np.int32)      #Should be DOUBLE
        
        else:       #Add the data
            existingVerNum = self.vertex_data.shape[0]
            input_faces_ += existingVerNum
            input_faces_ = input_faces_.astype(np.int32)
            self.meshindex_data = np.concatenate( (self.meshindex_data, input_faces_), axis=0)  #(Vx3,)
            # self.meshindex_data = self.meshindex_data.astype(np.int32)      #Should be DOUBLE

            self.vertex_data = np.concatenate( (self.vertex_data, input_vertices), axis=0)    #(6870,3)
            self.normal_data =  np.concatenate( (self.normal_data, input_normal[0]), axis=0)     #(6870,3)

            if color is None:
                color_data = np.ones(input_vertices.shape)   #(6870,3)
            else:
                color_data = np.tile(color, (input_vertices.shape[0], 1) )   #(6870,3)
            
            self.color_data =  np.concatenate( (self.color_data, color_data), axis=0)     #(6870,3)
            
            
        self.vertex_dim = self.vertex_data.shape[1]
        self.n_vertices = self.vertex_data.shape[0]

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.vertex_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.normal_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.color_data, GL_STATIC_DRAW)


        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.meshindex_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)


    def drawMesh(self):
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

        # # Handle color buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, self.n_vertices)

        glDisableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glUseProgram(0)

    def display(self):
        # First we draw a scene.
        # Notice the result is stored in the texture buffer.
        # Then we return to the default frame buffer since we will display on the screen.
        # glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        #Some anti-aliasing code (seems not working, though)
        if self.bAntiAliasing:
            glEnable (GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable (GL_LINE_SMOOTH)
            glHint (GL_LINE_SMOOTH_HINT, GL_NICEST)
            glEnable(GL_POLYGON_SMOOTH)
            glEnable(GL_MULTISAMPLE)
            # glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST)
        else:
            glDisable(GL_BLEND)
            glDisable(GL_MULTISAMPLE)

        # Set up viewing transformation, looking down -Z axis
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        #gluLookAt(0, 0, -g_fViewDistance, 0, 0, 0, -.1, 0, 0)   #-.1,0,0
        gluLookAt(0,0,0, 0, 0, 1, 0, -1, 0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        #gluPerspective(zoom, float(g_Width)/float(g_Height), g_nearPlane, g_farPlane)
        gluPerspective(65, float(self.width)/float(self.height), 10, 6000)         # This should be called here (not in the reshpe)
        # glMatrixMode(GL_MODELVIEW)
        # Render the scene

        if self.m_viewmode =="cam":
            if self.bOrthoCam:
                self.setCameraViewOrth()
            else:
                self.setCameraView()

        elif self.m_viewmode =="side":
            # self.setCameraViewOrth()
            self.setSideView()
            # self.setFree3DView()
            # 
        elif self.m_viewmode =="free":
            self.setFree3DView()
        else:
            assert False
            
        glClearColor(1.0, 1.0, 1.0, 1.0)
        # glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    
        # self.RenderDomeFloor()
        # glColor3f(0,1,0)
        # glTranslated(500,0,0)
        # glutSolidSphere(10, 10, 200)
        # glPointSize(30)
        # glBegin(GL_POINTS)
        # # glVertex3f(0,0,0)
        # if self.m_meshCenter is not None:
        #     glVertex3fv(self.m_meshCenter)
        # glEnd()
        # glutSolidTeapot(100, 10, 10)

        if self.bShowBackground:
            if self.bOrthoCam:
                self.drawBackgroundOrth()
            else:
                self.drawBackgroundPersp()

        if self.bShowFloor:
            self.drawFloor()

        self.drawMesh()

        glutSwapBuffers()
        glutPostRedisplay()

    def offscreenMode(self, bTrue):
        self.bOffscreenMode = bTrue

    def show_once(self):
        # for i in [1,2,3,4,5]:
        glutPostRedisplay()
        glutMainLoopEvent()


    def show(self):
        glutMainLoop()


# Just for an example
def loadSMPL():
    from modelViewer.batch_smpl import SMPL

    smplWrapper = SMPL("/home/hjoo/codes/glViewer/models/neutral_smpl_with_cocoplus_reg.pkl")

    #beta = np.array([ 0.60774645,  0.76093562, -0.46162634,  0.0948126 ,  0.05115048, 0.18798076,  0.02297921, -0.2294132 ,  0.14359247,  0.07730228])
    beta = np.array([-0.2723351 ,  0.24493244,  0.66188693,  3.080746  ,  1.803318  ,-0.06044015, -0.19139446,  0.07565568,  0.9439081 , -0.51000655])

    pose = np.array([ 1.21531341,  1.11861894,  1.1508407 ,  0.03090198,  0.07568664,
            0.05786929, -0.01946101, -0.04791601,  0.00566624, -0.01975956,
            0.04040587, -0.02909228,  0.17217339, -0.18295232, -0.05333628,
            0.24321426,  0.16652959,  0.01652244,  0.184938  , -0.08139203,
            0.08136881, -0.09354749,  0.22522661, -0.07165985, -0.08359848,
            -0.27758324,  0.00502749, -0.17570865, -0.00369029, -0.0219912,
            -0.34913435, -0.05382582,  0.22288936,  0.10101145,  0.32377259,
            -0.08444951, -0.03223499, -0.07053581, -0.08183003, -0.1110253 ,
            0.00895658, -0.38919476, -0.00748763, -0.02522146,  0.5864923 ,
            0.58635307, -0.00583143, -0.03246076, -0.10047339, -0.92346576,
            -0.36538482,  0.2815331 ,  0.24593229,  0.79902594, -0.17193863,
            -2.14336745,  0.39068873, -0.15159283,  0.2525081 , -0.02509047,
            0.08939309, -0.0801741 ,  0.40276617, -0.03815543, -0.05893454,
            -0.07858882, -0.24278936, -0.3096408 , -0.55118646, -0.09647344,
            0.45875036,  0.42067384])

    pose = np.array([ 9.0558e-01,  6.4592e-01, -2.8690e+00,  3.1094e-01,  1.0175e-01,
         5.6915e-02,  3.7163e-01, -1.1514e-01, -3.2411e-02,  9.1940e-02,
         1.3573e-02,  5.8944e-03, -3.4365e-02, -1.7157e-01,  2.0417e-02,
        -6.8286e-02,  1.2189e-03, -2.1876e-02,  1.2365e-01, -6.9564e-02,
         4.4505e-03, -6.7063e-02,  1.0760e-01,  7.0232e-02, -1.2466e-01,
        -1.3891e-01, -1.2108e-01, -1.6219e-02, -5.5884e-02, -9.7147e-03,
        -4.0098e-02,  1.6649e-01, -1.4749e-01,  1.7493e-01, -3.9301e-02,
         2.2233e-01,  2.2567e-01, -1.9609e-01,  1.3878e-02,  1.2296e-01,
         7.6158e-04, -5.5521e-01,  5.7593e-02,  7.3970e-02,  5.6500e-01,
         7.9010e-02, -2.0025e-01, -3.3629e-02,  4.0182e-02, -1.7911e-01,
        -8.7417e-01,  1.4417e-01,  1.3365e-01,  9.1869e-01,  2.1439e-01,
        -2.9541e-01,  8.5324e-02, -5.2092e-02,  2.0730e-01, -1.1425e-01,
        -6.1498e-02,  7.6002e-02, -2.3677e-01, -3.5770e-02, -7.9627e-02,
         1.5318e-01, -1.4370e-01, -4.8611e-02, -1.3202e-01, -6.9166e-02,
         1.3943e-01,  1.9009e-01])
    
    # pose = np.array([ 2.2487712, 0.28050578,-2.1792502 , -0.10493116,  0.01239435,  0.02972716,  0.08953293,-0.10654334, -0.00504329,  0.1593982 ,  0.01969572,  0.08852343, 0.09914812,  0.12574932, -0.02512331, -0.01473788, -0.04562924, 0.04665173,  0.0474331 , -0.0616711 , -0.00967203,  0.05010046, 0.1775912 , -0.08904129, -0.06684269, -0.14769007,  0.10105508, 0.0688806 , -0.02561731,  0.00964942, -0.1680568 ,  0.14983022, 0.20799895,  0.06796098,  0.10919931, -0.20863819,  0.00823393,-0.17863278,  0.09926094,  0.01495223, -0.08837841, -0.28607178,-0.11105742,  0.24558525,  0.06441574,  0.299364  , -0.15079273, 0.02175152,  0.20322715, -0.45768845, -0.9899641 , -0.06223915, 0.5227556 ,  0.6171622 ,  0.1368894 , -1.3889741 ,  0.19389033,-0.24303943,  1.1106223 , -0.2655932 , -0.6844785 , -0.17720126,-0.1870633 , -0.30705413,  0.08231031,  0.1118647 ,  0.02531371, 0.00614487, -0.05623743, -0.01657844,  0.07361342,  0.04853413])

    """Converting SMPL parameters to vertices"""
    beta = beta[np.newaxis,:]
    pose = pose[np.newaxis,:]
    v, j,_  = smplWrapper(beta, pose)  #beta: (N,10), pose: (N, 72)... return: v:(N,6890,3), j(N,19,3)
    v *=100
    j *=100


    #Load densepose data
    import scipy.io as sio
    densepose_info = sio.loadmat('/home/hjoo/codes/glViewer/densepose_uv_data/UV_Processed.mat')      #All_FaceIndices (13774), All_Faces(13774), All_U(7829), All_U_norm(7829), All_V(7829), All_V_norm (7829), All_vertices (7829)
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

    # faces = smplWrapper.f
    v =v[0]  #(6890,3)
    dp_vertex = v[densepose_info['All_vertices']-1]  #(1,7829,3)        #Considering repeatation
    faces =densepose_info['All_Faces']-1 #0~7828
    # vertexColor = densepose_info['All_FaceIndices']     #(13774,1)
    # vertexColor = np.repeat(vertexColor,3,axis=1) /24.0   #(13774,3)

    # vertexColor = densepose_info['All_U_norm']     #(13774,1)
    vertexColor = densepose_info['All_V_norm']     #(13774,1)
    vertexColor =  np.repeat(vertexColor,3,axis=1) 

    # vertexColor[vertexColor!=2]*=0
    # vertexColor[vertexColor==2]=24
    return dp_vertex, faces, vertexColor


if __name__ == '__main__':
    import cv2
    import viewer2D

    render = glRender()
    #load smpl data
    v, f, color= loadSMPL()
    render.set_mesh(v[0],f, color)

    if False:
        render.show()
    else:
        render.display()
        out_all_f = render.get_screen_color()
        # out_all_f = render.get_z_value()
        out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGRA)
        viewer2D.ImShow(out_all_f,waitTime=0)


        out_all_f = render.get_screen_color(GL_BACK)
        # out_all_f = render.get_z_value()
        out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGRA)
        viewer2D.ImShow(out_all_f,waitTime=0)

        cv2.imwrite('/home/hjoo/temp/render_general/test.jpg', out_all_f*255.0)


