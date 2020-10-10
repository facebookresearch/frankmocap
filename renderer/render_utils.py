# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np


# vertices: frames x meshVerNum x 3
# trifaces: facePolygonNum x 3 = 22800 x 3
def ComputeNormal(vertices, trifaces):

    if vertices.shape[0] > 5000:
        print('ComputeNormal: Warning: too big to compute {0}'.format(vertices.shape) )
        return

    #compute vertex Normals for all frames
    U = vertices[:,trifaces[:,1],:] - vertices[:,trifaces[:,0],:]  #frames x faceNum x 3
    V = vertices[:,trifaces[:,2],:] - vertices[:,trifaces[:,1],:]  #frames x faceNum x 3
    originalShape = U.shape  #remember: frames x faceNum x 3

    U = np.reshape(U, [-1,3])
    V = np.reshape(V, [-1,3])
    faceNormals = np.cross(U,V)     #frames x 13776 x 3
    from sklearn.preprocessing import normalize

    if np.isnan(np.max(faceNormals)):
        print('ComputeNormal: Warning nan is detected {0}')
        return
    faceNormals = normalize(faceNormals)

    faceNormals = np.reshape(faceNormals, originalShape)

    if False:        #Slow version
        vertex_normals = np.zeros(vertices.shape) #(frames x 11510) x 3
        for fIdx, vIdx in enumerate(trifaces[:,0]):
            vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
        for fIdx, vIdx in enumerate(trifaces[:,1]):
            vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
        for fIdx, vIdx in enumerate(trifaces[:,2]):
            vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
    else:   #Faster version
        # Computing vertex normals, much faster (and obscure) replacement
        index = np.vstack((np.ravel(trifaces), np.repeat(np.arange(len(trifaces)), 3))).T
        index_sorted = index[index[:,0].argsort()]
        vertex_normals = np.add.reduceat(faceNormals[:,index_sorted[:, 1],:][0],
            np.concatenate(([0], np.cumsum(np.unique(index_sorted[:, 0],
            return_counts=True)[1])[:-1])))[None, :]
        vertex_normals = vertex_normals.astype(np.float64)

    originalShape = vertex_normals.shape
    vertex_normals = np.reshape(vertex_normals, [-1,3])
    vertex_normals = normalize(vertex_normals)
    vertex_normals = np.reshape(vertex_normals,originalShape)

    return vertex_normals



def ComputeNormal_gpu(vertices, trifaces):
    import torch
    import torch.nn.functional as F

    if vertices.shape[0] > 5000:
        print('ComputeNormal: Warning: too big to compute {0}'.format(vertices.shape) )
        return

    #compute vertex Normals for all frames
    #trifaces_cuda = torch.from_numpy(trifaces.astype(np.long)).cuda()
    vertices_cuda = torch.from_numpy(vertices.astype(np.float32)).cuda()

    U_cuda = vertices_cuda[:,trifaces[:,1],:] - vertices_cuda[:,trifaces[:,0],:]  #frames x faceNum x 3
    V_cuda = vertices_cuda[:,trifaces[:,2],:] - vertices_cuda[:,trifaces[:,1],:]  #frames x faceNum x 3
    originalShape = list(U_cuda.size())  #remember: frames x faceNum x 3

    U_cuda = torch.reshape(U_cuda, [-1,3])#.astype(np.float32)
    V_cuda = torch.reshape(V_cuda, [-1,3])#.astype(np.float32)

    faceNormals = U_cuda.cross(V_cuda)
    faceNormals = F.normalize(faceNormals,dim=1)

    faceNormals = torch.reshape(faceNormals, originalShape)

    # trifaces has duplicated vertex index, so cannot be parallazied
    # vertex_normals = torch.zeros(vertices.shape,dtype=torch.float32).cuda() #(frames x 11510) x 3
    # for fIdx, vIdx in enumerate(trifaces[:,0]):
    #    vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
    # for fIdx, vIdx in enumerate(trifaces[:,1]):
    #     vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]
    # for fIdx, vIdx in enumerate(trifaces[:,2]):
    #     vertex_normals[:,vIdx,:] += faceNormals[:,fIdx,:]

    # Computing vertex normals, much faster (and obscure) replacement
    index = np.vstack((np.ravel(trifaces), np.repeat(np.arange(len(trifaces)), 3))).T
    index_sorted = index[index[:,0].argsort()]
    vertex_normals = np.add.reduceat(faceNormals[:,index_sorted[:, 1],:][0],
        np.concatenate(([0], np.cumsum(np.unique(index_sorted[:, 0],
        return_counts=True)[1])[:-1])))[None, :]
    vertex_normals = torch.from_numpy(vertex_normals).float().cuda()

    vertex_normals = F.normalize(vertex_normals,dim=2)
    vertex_normals = vertex_normals.data.cpu().numpy()  #(batch, chunksize, dim)

    return vertex_normals
