# Copyright (c) Facebook, Inc. and its affiliates.

#Visualization Function

import cv2

import numpy as np
import PIL
from PIL.Image import Image

def __ValidateNumpyImg(inputImg):
    if isinstance(inputImg, Image):
        # inputImg = cv2.cvtColor(np.array(inputImg), cv2.COLOR_RGB2BGR)
        inputImg = np.array(inputImg)

    return inputImg     #Q? is this copying someting (wasting memory or time?)?

veryFirstImShow = True
def ImShow(inputImg, waitTime=1, bConvRGB2BGR=False,name='image', scale=1.0):

    inputImg = __ValidateNumpyImg(inputImg)

    if scale!=1.0:
        inputImg = cv2.resize(inputImg, (inputImg.shape[0]*int(scale), inputImg.shape[1]*int(scale)))


    if bConvRGB2BGR:
        inputImg = cv2.cvtColor(inputImg, cv2.COLOR_RGB2BGR)

    cv2.imshow(name,inputImg)

    global veryFirstImShow
    if False:#veryFirstImShow:
        print(">> Press any key to move on")
        cv2.waitKey(0)      #the initial one is always blank... why?
        veryFirstImShow = 0
    else:
        cv2.waitKey(waitTime)

def ImgSC(inputImg, waitTime=1, bConvRGB2BGR=False,name='image', scale=1.0):

    inputImg = __ValidateNumpyImg(inputImg)

    minVal = np.min(inputImg)
    maxVal = np.max(inputImg)

    #rescale 
    inputImg = (inputImg-minVal)/ (maxVal-minVal)*255

    if scale!=1.0:
        inputImg = cv2.resize(inputImg, (inputImg.shape[0]*int(scale), inputImg.shape[1]*int(scale)))


    if bConvRGB2BGR:
        inputImg = cv2.cvtColor(inputImg, cv2.COLOR_RGB2BGR)

    cv2.imshow(name,inputImg)

    global veryFirstImShow
    if veryFirstImShow:
        print(">> Press any key to move on")
        cv2.waitKey(0)      #the initial one is always blank... why?
        veryFirstImShow = 0
    else:
        cv2.waitKey(waitTime)


# import matplotlib.pyplot as plt
# def Plot(values, title=None):
#     plt.plot(values)

#     if title is not None:
#         plt.title(title)#, loc='left', fontsize=12, fontweight=0, color='orange')

#     plt.show()

#bbe: min_pt, max_pt
def Vis_Bbox_minmaxPt(inputImg, min_pt, max_pt, color=None):

    bbr = [min_pt[0],min_pt[1], max_pt[0]- min_pt[0], max_pt[1]- min_pt[1]]
    return Vis_Bbox(inputImg, bbr, color)


def Vis_Bbox_XYXY(inputImg, bbox_xyxy, color=None):

    #draw biggest bbox
    pt1 = ( int(bbox_xyxy[0]),int(bbox_xyxy[1]) )
    pt2 = (int(bbox_xyxy[2]),int(bbox_xyxy[3]) )

    if color is None:
        color = (0,0,255)
    cv2.rectangle(inputImg, pt1, pt2,color, 3)

    return inputImg



def Vis_Bbox(inputImg, bbox_xyhw, color= None):
    return Vis_Bbox_XYWH(inputImg, bbox_xyhw, color)

#bbe: [leftTop_x,leftTop_y,width,height]
def Vis_Bbox_XYWH(inputImg, bbox_xyhw, color= None):

    inputImg = __ValidateNumpyImg(inputImg)

    #draw biggest bbox
    pt1 = ( int(bbox_xyhw[0]),int(bbox_xyhw[1]) )
    pt2 = (int(bbox_xyhw[0]  + bbox_xyhw[2]),int(bbox_xyhw[1] + bbox_xyhw[3]) )

    if color is None:
        color = (0,0,255)
    cv2.rectangle(inputImg, pt1, pt2,color, 3)

    return inputImg


def Vis_CocoBbox(inputImg, coco_annot):

    inputImg = __ValidateNumpyImg(inputImg)

    bbr =  np.round(coco_annot['bbox'])  #[leftTop_x,leftTop_y,width,height]

    #draw biggest bbox
    pt1 = ( int(bbr[0]),int(bbr[1]) )
    pt2 = (int(bbr[0]  + bbr[2]),int(bbr[1] + bbr[3]) )
    cv2.rectangle(inputImg, pt1, pt2,(255,255,255), 3)

    return inputImg


# connections_right = [
#                     {0, 2}, {2, 4}, {0, 6}	//nect, rightEye, rightEar
#                         , {6, 8}, {8, 10}, {6,12}, {12,14} , {14, 16}
#                             };
#]


def Vis_CocoSkeleton(keypoints, image=None):
# def Vis_CocoSkeleton(inputImg, coco_annot):

    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    image = __ValidateNumpyImg(image)

    #COCO17 original annotation ordering
    link2D = [ [0, 1], [1,3],		#nose(0), leftEye(1), leftEar(3)
                [0,5], [5, 7], [7, 9],			#leftShoulder(5), leftArm(7), leftWrist(9)
        	    [0, 11], [11, 13], [13, 15],  #leftHip(11), leftKnee(13), leftAnkle(15)

                [0,2], [2,4],     #nose(0), rightEye(2), rightEar(4)
                [0,6], [6, 8], [8, 10],			#rightShoulder(6), rightArm(8), rightWrist(10)
                [0, 12], [12, 14], [14, 16]  #rightHip(12), rightKnee(14), rightAnkle(16)
                ]

    bLeft = [ 1,1,
             1, 1, 1,
              1,1,1,
              0,0,
              0,0,0,
               0,0,0]

    # keypoints =  np.round(coco_annot['keypoints'])  #coco_annot['keypoints']: list with length 51
    if keypoints.shape[0] == 51:
        keypoints = np.reshape(keypoints, (-1,3))   #(17,3): (X, Y, Label)
    else:
        keypoints = np.reshape(keypoints, (-1,2))   #(17,3): (X, Y, Label)

    radius = 4

    for k in np.arange( len(keypoints) ):
        cv2.circle(image, (int(keypoints[k][0]), int(keypoints[k][1]) ), radius,(0,0,255),-1)

    for k in np.arange( len(link2D) ):
        parent = link2D[k][0]
        child = link2D[k][1]
        if bLeft[k]:
            c = (0,0,255)#BGR, RED
        else:
            c = (200,200,200)

        if keypoints[parent][0] ==0 or keypoints[child][0]==0: #		//not annotated one
            continue

        cv2.line(image, (int(keypoints[parent][0]), int(keypoints[parent][1])), (int(keypoints[child][0]), int(keypoints[child][1])), c, radius - 2)

    return image

DP_partIdx ={
    'Torso_Back': 1,
    'Torso_Front': 2,
    'RHand': 3,
    'LHand': 4,
    'LFoot': 5,
    'RFoot': 6,

    'R_upperLeg_back': 7,
    'L_upperLeg_back': 8,
    'R_upperLeg_front': 9,
    'L_upperLeg_front': 10,

    'R_lowerLeg_back': 11,
    'L_lowerLeg_back': 12,
    'R_lowerLeg_front': 13,
    'L_lowerLeg_front': 14,

    'L_upperArm_front': 15,
    'R_upperArm_front': 16,
    'L_upperArm_back': 17,
    'R_upperArm_back': 18,

    'L_lowerArm_back': 19,
    'R_lowerArm_back': 20,
    'L_lowerArm_front': 21,
    'R_lowerArm_front': 22,

    'RFace': 23,
    'LFace': 24
}
def Vis_Densepose(inputImg, coco_annot):

    inputImg = __ValidateNumpyImg(inputImg)

    import sys
    sys.path.append('/home/hjoo/data/DensePose/detectron/utils/')
    import densepose_methods as dp_utils
    DP = dp_utils.DensePoseMethods()

    if('dp_x' not in coco_annot.keys()):
        print("## Warning: No Densepose coco_annotation")
        return inputImg

    bbr =  np.round(coco_annot['bbox'])  #[leftTop_x,leftTop_y,width,height]

    Point_x = np.array(coco_annot['dp_x'])/ 255. * bbr[2] + bbr[0] # Strech the points to current box. from 255x255 -> [bboxWidth,bboxheight]
    Point_y = np.array(coco_annot['dp_y'])/ 255. * bbr[3] + bbr[1] # Strech the points to current box.
    # part_seg_index = np.array(coco_annot['dp_I'])   # part segment info

    #coco_annot['dp_I']: indexing
    # Torso Back: 1
    # Torso front: 2
    # RHand: 3
    # LHand: 4
    # LFoot: 5
    # RFoot: 6

    # R_upperLeg_back 7
    # L_upperLeg_back 8
    # R_upperLeg_front 9
    # L_upperLeg_front 10

    # R_lowerLeg_back 11
    # L_lowerLeg_back 12
    # R_lowerLeg_front 13
    # L_lowerLeg_front 14

    # L_upperArm_front 15
    # R_upperArm_front 16
    # L_upperArm_back 17
    # R_upperArm_back 18

    # L_lowerArm_back 19
    # R_lowerArm_back 20
    # L_lowerArm_front 21
    # R_lowerArm_front 22

    # RFace: 23
    # LFace: 24

    #Found BBoxes for rhand, lhand, and face using DensePose Data
    RHandIdx = [i for i,x in enumerate(coco_annot['dp_I']) if x == DP_partIdx['RHand'] ] #3.0]
    if len(RHandIdx)>0:

        minX = min(Point_x[RHandIdx])
        maxX = max(Point_x[RHandIdx])
        minY = min(Point_y[RHandIdx])
        maxY = max(Point_y[RHandIdx])
        RhandBBox = [minX, minY, maxX-minX, maxY-minY]
    else:
        RhandBBox = [-1,-1,-1,-1]

    LHandIdx = [i for i,x in enumerate(coco_annot['dp_I']) if x == DP_partIdx['LHand'] ]#4.0]
    if len(LHandIdx)>0:
        minX = min(Point_x[LHandIdx])
        maxX = max(Point_x[LHandIdx])
        minY = min(Point_y[LHandIdx])
        maxY = max(Point_y[LHandIdx])
        LhandBBox = [minX, minY, maxX-minX, maxY-minY]
    else:
        LhandBBox = [-1,-1,-1,-1]

    FaceIdx = [i for i,x in enumerate(coco_annot['dp_I']) if x == DP_partIdx['RFace'] or x == DP_partIdx['LFace'] ]   #23.0 or x == 24.0]
    if len(FaceIdx)>0:
        minX = min(Point_x[FaceIdx])
        maxX = max(Point_x[FaceIdx])
        minY = min(Point_y[FaceIdx])
        maxY = max(Point_y[FaceIdx])
        FaceBBox = [minX, minY, maxX-minX, maxY-minY]
    else:
        FaceBBox = [-1,-1,-1,-1]

    # #U,V,I -> Adam vertex  (Todo: should be reverified)
    # adamVerIdx_vec = np.zeros(len(coco_annot['dp_I']))
    # for i, (ii,uu,vv) in enumerate(zip(coco_annot['dp_I'],coco_annot['dp_U'],coco_annot['dp_V'])):

    #     vertexId = DP.IUV2VertexId(ii,uu,vv)
    #     adamVerIdx_vec[i] = vertexId

    # #draw biggest bbox
    # pt1 = ( int(bbr[0]),int(bbr[1]) )
    # pt2 = (int(bbr[0]  + bbr[2]),int(bbr[1] + bbr[3]) )
    # cv2.rectangle(inputImg, pt1, pt2,(0,0,0),1)

    #draw RHand bbox
    pt1 = ( int(RhandBBox[0]),int(RhandBBox[1]) )
    pt2 = (int(RhandBBox[0]  + RhandBBox[2]),int(RhandBBox[1] + RhandBBox[3]) )
    cv2.rectangle(inputImg, pt1, pt2,(0,0,255),2)

    #draw lHand bbox
    pt1 = ( int(LhandBBox[0]),int(LhandBBox[1]) )
    pt2 = (int(LhandBBox[0]  + LhandBBox[2]),int(LhandBBox[1] + LhandBBox[3]) )
    cv2.rectangle(inputImg, pt1, pt2,(0,255,0),2)

    #draw Face bbox
    pt1 = ( int(FaceBBox[0]),int(FaceBBox[1]) )
    pt2 = (int(FaceBBox[0]  + FaceBBox[2]),int(FaceBBox[1] + FaceBBox[3]) )
    cv2.rectangle(inputImg, pt1, pt2,(255,0,0),2)

    # Draw Densepose Keypoints
    tempColorIdx  = np.array(coco_annot['dp_I'])/ 24 *255
    #tempColorIdx  = np.array(coco_annot['dp_U']) *255
    #tempColorIdx  = np.array(coco_annot['dp_V']) *255
    tempColorIdx = np.uint8(tempColorIdx)
    tempColorIdx = cv2.applyColorMap(tempColorIdx, cv2.COLORMAP_JET)
    for cnt, pt in enumerate(zip(Point_x,Point_y,tempColorIdx, coco_annot['dp_I'])):
        # if pt[3] != DP_partIdx['Torso_Front']: #Uncomment this if you want to draw specific part
        #     continue
        #tempColorIdx  = coco_annot['dp_I']
        tempColor = pt[2][0].astype(np.int32).tolist()
        cv2.circle(inputImg,(int(pt[0]),int(pt[1])), 5,tempColor, -1)

    return inputImg


#H36m skeleton32
#skel can be
# : (17,2)       #with Nose
# : (16,2)       #without NOse
# : (32,2)       #original
def Vis_Skeleton_2D_H36m(pt2d, image = None, color=None):
    pt2d = np.reshape(pt2d,[-1,2])          #Just in case. Make sure (32, 2)


    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4

    if(pt2d.shape[0]==16):
        print("Vis_Skeleton_2D_H36m: {} joints".format(16))
        #Without Nose
        link2D = [ [0,1],[1,2],[2,3],#root(0), rHip(1), rKnee(2), rAnkle(3)
                            [0,4],[4,5],[5,6],#root(0, lHip(4), lKnee(5), lAnkle(6)
                            [0,7], [7,8], [8,9], #root(0, spineMid(7), neck(8), head(9)
                            [8,10], [10,11], [11,12], #Left Arms. neck(8). lshoulder(10),  lElbow(11), lWrist (12)
                            [8,13], [13,14], [14,15] #Right Arm, neck(8), rshoulder(13),  rElbow(14), rWrist (15)
                            ]
        bLeft = [ 0,0,0,
            1, 1, 1,
            1,1,1,
            1,1,1,
            0,0,0]
    elif pt2d.shape[0]==17:
        print("Vis_Skeleton_2D_H36m: {} joints".format(17))
        #With Nose
        link2D = [ [0,1],[1,2],[2,3],#root(0), rHip(1), rKnee(2), rAnkle(3)
                            [0,4],[4,5],[5,6],#root(0, lHip(4), lKnee(5), lAnkle(6)
                            [0,7], [7,8], [8,9], [9,10], #root(0, spineMid(7), neck(8), nose(9), head(9)
                            [8,11], [11,12], [12,13], #Left Arms. neck(8). lshoulder(11),  lElbow(12), lWrist (13)
                            [8,14], [14,15], [15,16] #Right Arm, neck(8), rshoulder(14),  rElbow(15), rWrist (16)
                            ]
        bLeft = [ 0,0,0,
            1, 1, 1,
            1,1,1, 1,
            1,1,1,
            0,0,0]
    else:
        print("Vis_Skeleton_2D_H36m: {} joints".format(32))
        #Human 36m DB's mocap data. 32 joints
        link2D = [ [0,1],[1,2],[2,3],[3,4],[4,5], #RightLeg: root(0), rHip(1), rKnee(2), rAnkle(3), rFootMid(4), rFootEnd(5)
                     [0,6],[6,7],[7,8],[8,9], [9,10], #LeftLeg: root, lHip(6), lKnee(7), lAnkle(8), lFootMid(9), lFootEnd(10)
                     [11,12], [12,13], [13,14], [14,15], #root2(11), spineMid(12), neck(13), nose(14), head(15) #0,11 are the same points?
                     [16,17], [17,18], [18,19], [20,21], [20,22],   #Left Arms. neck(16==13), lshoulder(17),  lElbow(18), lWrist (19=20), lThumb(21), lMiddleFinger(22)
                     [24,25], [25,26], [26,27], [27,29], [27,30]   #Right Arm, neck(24==13), rshoulder(25),  rElbow(26), rWrist (27=28), rThumb(29), rMiddleFinger(30)
                     ]
        bLeft = [0 ,0, 0, 0, 0,
                    1, 1, 1, 1, 1,
                    1, 1, 1, 1,
                    1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0] #To draw left as different color. Torso is treated as left


    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    for k in np.arange( len(link2D) ):
        parent = link2D[k][0]
        child = link2D[k][1]
        if color is not None:
            c = color
        else:
            if bLeft[k]:
                c = (0,0,255)#BGR, RED
            else:
                c = (0,0,0)

        cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    return image




#Panoptic Studio SMC19 ordering

def Vis_Skeleton_2D_SMC19(pt2d, image = None, color=None):
    pt2d = np.reshape(pt2d,[-1,2])          #Just in case. Make sure (32, 2)


    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4

    assert pt2d.shape[0]==19
    print("Vis_Skeleton_2D_H36m: {} joints".format(16))
    #Without Nose
    link2D = [ [0,1], [0,2],  #neck -> nose, neck-> bodyCenter
                 [0,3], [3,4], [4,5],   #Left Arm

                 [2,6], [6,7], [7,8],   #left leg
                [2,12],[12,13], [13,14], #Right leg
                [0,9], [9, 10], [10, 11], #Right Arm
                [1, 15], [15, 16], #left eye
                [1, 17], [17, 18]] #right eye

    bLeft = [ 1,1,
        1, 1, 1,
        1,1,1,
        0,0,0,
        0,0,0,
        1,1,
        0,0]


    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    for k in np.arange( len(link2D) ):
        parent = link2D[k][0]
        child = link2D[k][1]
        if color is not None:
            c = color
        else:
            if bLeft[k]:
                c = (0,0,255)#BGR, RED
            else:
                c = (0,0,0)

        cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    return image




#Panoptic Studio SMC19 ordering
 
def Vis_Skeleton_2D_SMC19(pt2d, image = None, color=None):
    pt2d = np.reshape(pt2d,[-1,2])          #Just in case. Make sure (32, 2)


    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255        

    radius = 4

    assert pt2d.shape[0]==19
    print("Vis_Skeleton_2D_H36m: {} joints".format(16))
    #Without Nose
    link2D = [ [0,1], [0,2],  #neck -> nose, neck-> bodyCenter
                 [0,3], [3,4], [4,5],   #Left Arm
                
                 [2,6], [6,7], [7,8],   #left leg
                [2,12],[12,13], [13,14], #Right leg
                [0,9], [9, 10], [10, 11], #Right Arm
                [1, 15], [15, 16], #left eye
                [1, 17], [17, 18]] #right eye
    
    bLeft = [ 1,1,
        1, 1, 1,
        1,1,1,
        0,0,0,
        0,0,0,
        1,1,
        0,0]

            
    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    for k in np.arange( len(link2D) ):
        parent = link2D[k][0]
        child = link2D[k][1]
        if color is not None:
            c = color
        else:
            if bLeft[k]:
                c = (0,0,255)#BGR, RED
            else:
                c = (0,0,0)

        cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    return image


def Vis_Skeleton_2D_Hand(pt2d, image = None, color=None):
    pt2d = np.reshape(pt2d,[-1,2])          #Just in case. Make sure (32, 2)

    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255        

    radius = 4

    # assert pt2d.shape[0]==19
    print("Vis_Skeleton_2D_H36m: {} joints".format(16))
    #Without Nose
    link2D = [ [0,1], [1,2], [2,3], [3,4],  #thumb
                [0,5], [5,6],[6,7],[7,8],   #index
                [0,9],[9,10],[10,11],[11,12],
                [0,13],[13,14],[14,15],[15,16],
                [0,17],[17,18],[18,19],[19,20]
                ]
            
    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    for k in np.arange( len(link2D) ):
        parent = link2D[k][0]
        child = link2D[k][1]
        if color is not None:
            c = color
        else:
            if True:#bLeft[k]:
                c = (0,255,255)#BGR, RED
            # else:
            #     c = (0,0,0)

        cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    return image





#smplCOCO19 or MTC20
#smplCOCO: (19,2)
#MTC: (20,2)
#pt2d: (36,)
#pt2d_visibility: (18,)
def Vis_Skeleton_2D_smplCOCO(pt2d, pt2d_visibility=None, image = None, blankImSize = 1000, bVis = False, color=None, bBoxWidth=None):
    pt2d = np.reshape(pt2d,[-1,2])          #Just in case. Make sure (32, 2)


    if pt2d_visibility is not None and len(pt2d_visibility) == len(pt2d)*2:
        pt2d_visibility = pt2d_visibility[::2]
    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((blankImSize,blankImSize,3),np.uint8) *255

    radius = 4

    if(pt2d.shape[0]==19 or pt2d.shape[0]==20):
        # print("Vis_Skeleton_2D_smplCOCO: {} joints".format(16))
        #Without Nose
        link2D = [ [12,2], [2,1], [1,0], #Right leg
                     [12,3], [3,4], [4,5], #Left leg
                     [12,9], [9,10], [10,11], #Left Arm
                     [12,8], [8,7], [7,6], #Right shoulder
                      [12,14],[14,16],[16,18],  #Neck(12)->Nose(14)->rightEye(16)->rightEar(18)
                      [14,15],[15,17],   #Nose(14)->leftEye(15)->leftEar(17).
                      [14,13] #Nose->headTop(13)
                     ]
        bLeft = [ 0,0,0,
            1, 1, 1,
            0,0,0,
            1,1,1,
            1,1,1,
            11,0,11,0]
    elif(pt2d.shape[0]==18): #No head (13)
        # print("Vis_Skeleton_2D_smplCOCO: {} joints".format(16))
        #Without Nose
        link2D = [ [12,2], [2,1], [1,0], #Right leg
                     [12,3], [3,4], [4,5], #Left leg
                     [12,9], [9,10], [10,11], #Left Arm
                     [12,8], [8,7], [7,6], #Right shoulder
                      [12,13],[13,15],[15,17],  #Neck(12)->Nose(14)->rightEye(16)->rightEar(18)
                      [13,14],[14,16]   #Nose(14)->leftEye(15)->leftEar(17).
                    #   [14,13] #Nose->headTop(13)
                     ]
        bLeft = [ 0,0,0,
            1, 1, 1,
            1,1,1,
            0,0,0,
            1,0,0,
            1,1]
    elif(pt2d.shape[0]==26): #SMPLCOCO totalCpa26
        #Without Nose
        link2D = [ [12,2], [2,1], [1,0], #Right leg
                    [12,3], [3,4], [4,5], #Left leg
                    [12,9], [9,10], [10,11], #Left Arm
                    [12,8], [8,7], [7,6], #Right shoulder
                    [12,14],[14,16],[16,18],  #Neck(12)->Nose(14)->rightEye(16)->rightEar(18)
                    [14,15],[15,17],   #Nose(14)->leftEye(15)->leftEar(17).
                    # [14,13], #Nose->headMidle(13)
                    [12,19],       #headTop19
                    [5,20], [5,21], [5,22],       #leftFoot
                    [0,23], [0,24], [0,25]       #rightFoot
                    ]
        bLeft = [ 0,0,0,
            1, 1, 1,
            1,1,1,
            0,0,0,
            1,0,0,
            1,1,
            1,
            1,1,1,
            0,0,0]

    else:
        assert False

    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        if pt2d_visibility is None:
            cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)
        else:
            if pt2d_visibility[k]:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    for k in np.arange( len(link2D) ):
        parent = link2D[k][0]
        child = link2D[k][1]
        if color is not None:
            c = color
        else:
            if bLeft[k]:
                c = (0,0,255)#BGR, RED
            else:
                c = (0,0,0)
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)


    if bBoxWidth is not None:
        image = Vis_Bbox_minmaxPt(image, [0,0], [bBoxWidth,bBoxWidth])

    if bVis:
        ImShow(image,name='Vis_Skeleton_2D_smplCOCO')
    return image


def Vis_Skeleton_2D_smpl24(pt2d, image = None, bVis = False, color=None):
    pt2d = np.reshape(pt2d,(-1,2))          #Just in case. Make sure (32, 2)

    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4


    #SMPL 24 joints used for LBS
    link2D = [ [0,3],[3,6],[6,9],[9,12],[12,15],  #root-> torso -> head
                    [9,13],[13,16],[16,18],[18,20],[20,22], #Nect-> left hand
                    [9,14], [14,17], [17,19], [19,21], [21,23],  #Nect-> right hand
                    [0,1], [1,4], [4,7], [7,10], # left Leg
                    [0,2], [2,5], [5,8], [8,11] #right leg
                    ]

    bLeft = [ 0,0,0,
        1, 1, 1,
        0,0,0,
        1,1,1,
        1,1,1,
        11,0,11,0]

    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    for k in np.arange( len(link2D) ):
        parent = link2D[k][0]
        child = link2D[k][1]
        if color is not None:
            c = color
        else:
            if True:#bLeft[k]:
                c = (0,0,255)#BGR, RED
            else:
                c = (0,0,0)

        cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    if bVis:
        ImShow(image)
    return image


def Vis_Skeleton_2D_smpl45(pt2d, image = None, bVis = False, color=None):
    pt2d = np.reshape(pt2d,(-1,2))          #Just in case. Make sure (32, 2)

    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4


    #SMPL 24 joints used for LBS
    link2D = [ [0,3],[3,6],[6,9],[9,12],[12,15],  #root-> torso -> head
                    [9,13],[13,16],[16,18],[18,20],[20,22], #Nect-> left hand
                    [9,14], [14,17], [17,19], [19,21], [21,23],  #Nect-> right hand
                    [0,1], [1,4], [4,7], [7,10], # left Leg
                    [0,2], [2,5], [5,8], [8,11] #right leg
                    ]

    bLeft = [ 0,0,0,
        1, 1, 1,
        0,0,0,
        1,1,1,
        1,1,1,
        11,0,11,0]

    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    for k in np.arange( len(link2D) ):
        parent = link2D[k][0]
        child = link2D[k][1]
        if color is not None:
            c = color
        else:
            if True:#bLeft[k]:
                c = (0,0,255)#BGR, RED
            else:
                c = (0,0,0)

        cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    if bVis:
        ImShow(image)
    return image




def Vis_Skeleton_2D_MPII(pt2d, pt2d_visibility = None, image = None, bVis = False, color=None):
    pt2d = np.reshape(pt2d,(-1,2))          #Just in case. Make sure (32, 2)

    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4
     #SMPL 24 joints used for LBS
    link2D = [ [6,7],[7,8],[8,9],  #root-> torso -> head
                    [7,12], [12,11],[11,10], #right arm
                    [7,13], [13,14], [14,15],  #left arm
                    [6,2],[2,1], [1,0], #right leg
                    [6,3], [3,4], [4,5] #left leg
                    ]

    bLeft = [ 1,1,1,
        0, 0, 0,
        1,1,1,
        0,0,0,
        1,1,1]
    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        if pt2d_visibility is None:
            cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)
        else:
            if pt2d_visibility[k]:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    for k in np.arange( len(link2D) ):
        parent = link2D[k][0]
        child = link2D[k][1]
        if color is not None:
            c = color
        else:
            if bLeft[k]:
                c = (0,0,255)#BGR, RED
            else:
                c = (0,0,0)
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    if bVis:
        ImShow(image)
    return image



def Vis_Skeleton_2D_foot(pt2d, pt2d_visibility = None, image = None, bVis = False, color=None):
    pt2d = np.reshape(pt2d,(-1,2))          #Just in case. Make sure (32, 2)

    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4
     #SMPL 24 joints used for LBS
    link2D = [ [0,1],[1,2],  #root-> torso -> head
                [3,4], [4,5] ]

    bLeft = [ 1,1,
        0, 0]
    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        if pt2d_visibility is None:
            cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)
        else:
            if pt2d_visibility[k]:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    for k in np.arange( len(link2D) ):
        parent = link2D[k][0]
        child = link2D[k][1]
        if color is not None:
            c = color
        else:
            if bLeft[k]:
                c = (0,0,255)#BGR, RED
            else:
                c = (0,0,0)
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    if bVis:
        ImShow(image)
    return image





def Vis_Skeleton_2D_Openpose25(pt2d, pt2d_visibility = None, image = None, bVis = False, color=None):
    pt2d = np.reshape(pt2d,(-1,2))          #Just in case. Make sure (32, 2)

    if pt2d.shape[0]==49:        #SPIN 25 (openpose) + 24 (superset) joint
        return Vis_Skeleton_2D_SPIN49(pt2d, pt2d_visibility, image, bVis, color)



    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4

    #Openpose25
    link_openpose = [  [8,1], [1,0] , [0,16] , [16,18] , [0,15], [15,17],
            [1,2],[2,3],[3,4],      #Right Arm
            [1,5], [5,6], [6,7],       #Left Arm
            [8,12], [12,13], [13,14], [14,21], [14,19], [14,20],
            [8,9], [9,10], [10,11], [11,24], [11,22], [11,23]
            ]

    bLeft = [ 1,1,1,1,0,0,
        0,0,0,
        1,1,1,
        1,1,1,1,1,1,
        0,0,0,0,0,0]



    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        if pt2d_visibility is None:
            cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)
        else:
            if pt2d_visibility[k]:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    #Openpose joint drawn as blue
    for k in np.arange( len(link_openpose) ):
        parent = link_openpose[k][0]
        child = link_openpose[k][1]
        if color is not None:
            c = color
        else:
            if bLeft[k]:
                c = (255,0,0)#BGR, Blue
            else:
                c = (0,0,0) #Right Black
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    return image


def Vis_Skeleton_2D_Openpose_hand(pt2d, pt2d_visibility = None, image = None, bVis = False, color=None):
    pt2d = np.reshape(pt2d,(-1,2))          #Just in case. Make sure (32, 2)

    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4

    #Openpose25
    link_openpose =  [ [0,1], [1,2], [2,3], [3,4],  #thumb
                    [0,5], [5,6],[6,7],[7,8],   #index
                    [0,9],[9,10],[10,11],[11,12],
                    [0,13],[13,14],[14,15],[15,16],
                    [0,17],[17,18],[18,19],[19,20]
                    ]
    link_openpose = np.array(link_openpose)


    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        if pt2d_visibility is None:
            cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)
        else:
            if pt2d_visibility[k]:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    #Openpose joint drawn as blue
    for k in np.arange( len(link_openpose) ):
        parent = link_openpose[k][0]
        child = link_openpose[k][1]
        if color is not None:
            c = color
        else:
            c = (255,0,0)#BGR, Blue
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    return image



def Vis_Skeleton_2D_Openpose18(pt2d, pt2d_visibility = None, image = None, bVis = False, color=None):
    pt2d = np.reshape(pt2d,(-1,2))          #Just in case. Make sure (32, 2)

    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4

    #Openpose18
    link_openpose = [    [1,0] , [0,14] , [14,16] , [0,15], [15,17],
                [1,2],[2,3],[3,4],      #Right Arm
                [1,5], [5,6], [6,7],       #Left Arm
                [1,11], [11,12], [12,13],       #Left Leg
                [8,1], [8,9], [9,10]       #Right Leg
                ]

    bLeft = [ 1,1,1,1,1,
        0,0,0,
        1,1,1,
        1,1,1,
        0,0,0]



    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        if pt2d_visibility is None:
            cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)
        else:
            if pt2d_visibility[k]:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    #Openpose joint drawn as blue
    for k in np.arange( len(link_openpose) ):
        parent = link_openpose[k][0]
        child = link_openpose[k][1]
        if color is not None:
            c = color
        else:
            if bLeft[k]:
                c = (255,0,0)#BGR, Blue
            else:
                c = (0,0,0)
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    return image



def Vis_Skeleton_2D_SPIN24(pt2d, pt2d_visibility = None, image = None, bVis = False, color=None):
    pt2d = np.reshape(pt2d,(-1,2))          #Just in case. Make sure (32, 2)

    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4



    #Openpose25 in Spin Defition + SPIN global 24
    # 'OP Nose', 'OP Neck', 'OP RShoulder',           #0,1,2
    # 'OP RElbow', 'OP RWrist', 'OP LShoulder',       #3,4,5
    # 'OP LElbow', 'OP LWrist', 'OP MidHip',          #6, 7,8
    # 'OP RHip', 'OP RKnee', 'OP RAnkle',             #9,10,11
    # 'OP LHip', 'OP LKnee', 'OP LAnkle',             #12,13,14
    # 'OP REye', 'OP LEye', 'OP REar',                #15,16,17
    # 'OP LEar', 'OP LBigToe', 'OP LSmallToe',        #18,19,20
    # 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',  #21, 22, 23, 24  ##Total 25 joints  for openpose
    link_openpose = [  [8,1], [1,0] , [0,16] , [16,18] , [0,15], [15,17],
                [1,2],[2,3],[3,4],      #Right Arm
                [1,5], [5,6], [6,7],       #Left Arm
                [8,12], [12,13], [13,14], [14,19], [19,20], [20,21],    #Left Leg
                [8,9], [9,10], [10,11], [11,22], [22,23], [23,24]       #Right left
                ]

    link_spin24 =[  [14,16], [16,12], [12,17] , [17,18] ,
                [12,9],[9,10],[10,11],      #Right Arm
                [12,8], [8,7], [7,6],       #Left Arm
                [14,3], [3,4], [4,5],
                [14,2], [2,1], [1,0]]


    link_spin24 = np.array(link_spin24) + 25

    # bLeft = [ 1,1,1,1,0,0,
    #     0,0,0,
    #     1,1,1,
    #     1,1,1,1,1,1,
    #     0,0,0,0,0,0]
    bLeft = [ 0,0,0,0,
        1,1,1,
        0,0,0,
        1,1,1,
        0,0,0]



    # for i in np.arange( len(link) ):
    for k in np.arange( 25,len(pt2d) ):
        if color is not None:
            if pt2d_visibility is None:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, color,-1)
            else:
                if pt2d_visibility[k]:
                    cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, color,-1)
        else:
            if pt2d_visibility is None:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,0,255),-1)
            else:
                if pt2d_visibility[k]:
                    cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,0,255),-1)

    # # #Openpose joint drawn as blue
    # for k in np.arange( len(link_openpose) ):
    #     parent = link_openpose[k][0]
    #     child = link_openpose[k][1]
    #     if color is not None:
    #         c = color
    #     else:
    #         if True:#bLeft[k]:
    #             c = (255,0,0)#BGR, Blue
    #         else:
    #             c = (0,0,0)
    #     if pt2d_visibility is None:
    #         cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
    #     else:
    #         if pt2d_visibility[parent] and pt2d_visibility[child]:
    #             cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)


    #SPIN24 joint drawn as red
    for k in np.arange( len(link_spin24) ):
        parent = link_spin24[k][0]
        child = link_spin24[k][1]
        if color is not None:
            c = color
        else:
            if True:#bLeft[k]:
                c = (0,0,255)#BGR, RED
            else:
                c = (0,0,0)
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)



    if bVis:
        ImShow(image)
    return image




def Vis_Skeleton_2D_SPIN49(pt2d, pt2d_visibility = None, image = None, bVis = False, color=None):
    pt2d = np.reshape(pt2d,(-1,2))          #Just in case. Make sure (32, 2)

    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4



    #Openpose25 in Spin Defition + SPIN global 24
    # 'OP Nose', 'OP Neck', 'OP RShoulder',           #0,1,2
    # 'OP RElbow', 'OP RWrist', 'OP LShoulder',       #3,4,5
    # 'OP LElbow', 'OP LWrist', 'OP MidHip',          #6, 7,8
    # 'OP RHip', 'OP RKnee', 'OP RAnkle',             #9,10,11
    # 'OP LHip', 'OP LKnee', 'OP LAnkle',             #12,13,14
    # 'OP REye', 'OP LEye', 'OP REar',                #15,16,17
    # 'OP LEar', 'OP LBigToe', 'OP LSmallToe',        #18,19,20
    # 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',  #21, 22, 23, 24  ##Total 25 joints  for openpose
    link_openpose = [  [8,1], [1,0] , [0,16] , [16,18] , [0,15], [15,17],
                [1,2],[2,3],[3,4],      #Right Arm
                [1,5], [5,6], [6,7],       #Left Arm
                [8,12], [12,13], [13,14], [14,19], [19,20], [20,21],    #Left Leg
                [8,9], [9,10], [10,11], [11,22], [22,23], [23,24]       #Right left
                ]

    link_spin24 =[  [14,16], [16,12], [12,17] , [17,18] ,
                [12,9],[9,10],[10,11],      #Right Arm
                [12,8], [8,7], [7,6],       #Left Arm
                [14,3], [3,4], [4,5],
                [14,2], [2,1], [1,0]]


    link_spin24 = np.array(link_spin24) + 25

    # bLeft = [ 1,1,1,1,0,0,
    #     0,0,0,
    #     1,1,1,
    #     1,1,1,1,1,1,
    #     0,0,0,0,0,0]
    bLeft = [ 0,0,0,0,
        1,1,1,
        0,0,0,
        1,1,1,
        0,0,0]



    # for i in np.arange( len(link) ):
    for k in np.arange( 25,len(pt2d) ):
        if color is not None:
            if pt2d_visibility is None:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, color,-1)
            else:
                if pt2d_visibility[k]:
                    cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, color,-1)
        else:
            if pt2d_visibility is None:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,0,255),-1)
            else:
                if pt2d_visibility[k]:
                    cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,0,255),-1)

    # #Openpose joint drawn as blue
    for k in np.arange( len(link_openpose) ):
        parent = link_openpose[k][0]
        child = link_openpose[k][1]
        if color is not None:
            c = color
        else:
            if True:#bLeft[k]:
                c = (255,0,0)#BGR, Blue
            else:
                c = (0,0,0)
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)


    #SPIN24 joint drawn as red
    for k in np.arange( len(link_spin24) ):
        parent = link_spin24[k][0]
        child = link_spin24[k][1]
        if color is not None:
            c = color
        else:
            if True:#bLeft[k]:
                c = (0,0,255)#BGR, RED
            else:
                c = (0,0,0)
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)



    if bVis:
        ImShow(image)
    return image




def Vis_Skeleton_2D_coco(pt2d, pt2d_visibility = None, image = None,  bVis = False, color=None , offsetXY =None):
    pt2d = np.reshape(pt2d,(-1,2))          #Just in case. Make sure (32, 2)

    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4

    # 'OP RHip', 'OP RKnee', 'OP RAnkle',             #9,10,11
    # 'OP LHip', 'OP LKnee', 'OP LAnkle',             #12,13,14
    # 'OP REye', 'OP LEye', 'OP REar',                #15,16,17
    # 'OP LEar', 'OP LBigToe', 'OP LSmallToe',        #18,19,20
    # 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',  #21, 22, 23, 24  ##Total 25 joints  for openpose
    link_coco = [  [0,1], [1,3] , [0,2] , [2,4],
                [6,8],[8,10],      #Right Arm
                [5,7], [7,9],       #Left Arm
                [15,13], [13,11], [11,5],    #Left Leg
                [16,14], [14,12], [12,6], #Right left
                ]

    for k in np.arange( len(pt2d) ):
        if color is not None:
            if pt2d_visibility is None:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, color,-1)
            else:
                if pt2d_visibility[k]:
                    cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, color,-1)
        else:
            if pt2d_visibility is None:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,0,255),-1)
            else:
                if pt2d_visibility[k]:
                    cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,0,255),-1)

    # # #Openpose joint drawn as blue
    for k in np.arange( len(link_coco) ):
        parent = link_coco[k][0]
        child = link_coco[k][1]
        if color is not None:
            c = color
        else:
            if True:#bLeft[k]:
                c = (255,0,0)#BGR, Blue
            else:
                c = (0,0,0)
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    if bVis:
        ImShow(image)
    return image




    

def Vis_Skeleton_2D_general(pt2d, pt2d_visibility = None, image = None,  bVis = False, color=None , offsetXY =None):
    pt2d = np.reshape(pt2d,(-1,2))          #Just in case. Make sure (32, 2)

    if offsetXY is not None:
        pt2d = pt2d + np.array(offsetXY)

    if pt2d.shape[0]==49:        #SPIN 25 (openpose) + 24 (superset) joint
        return Vis_Skeleton_2D_SPIN49(pt2d, pt2d_visibility, image, bVis, color)



    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4

    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    if bVis:
        ImShow(image)
    return image


def Vis_Skeleton_3Dto2D_general(pt2d, pt2d_visibility = None, image = None,  bVis = False, color=None, offsetXY =None):
    pt2d = np.reshape(pt2d,(-1,3))          #Just in case. Make sure (32, 2)


    if pt2d.shape[0]==49:        #SPIN 25 (openpose) + 24 (superset) joint
        return Vis_Skeleton_2D_SPIN49(pt2d, pt2d_visibility, image, bVis, color)

    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4

    # for i in np.arange( len(link) ):
    for k in np.arange( len(pt2d) ):
        cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,255,0),-1)

    if bVis:
        ImShow(image)
    return image

#H36m skeleton32
# def Vis_Skeleton_H36m16(inputImg, coco_annot):
