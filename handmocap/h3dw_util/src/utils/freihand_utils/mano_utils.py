import chumpy as ch
import numpy as np

kpId2vertices = {
                 4: [744],  #ThumbT
                 8: [320],  #IndexT
                 12: [443],  #MiddleT
                 16: [555],  #RingT
                 20: [672]  #PinkT
                 }


def get_keypoints_from_mesh_ch(mesh_vertices, keypoints_regressed):
    """ Assembles the full 21 keypoint set from the 16 Mano Keypoints and 5 mesh vertices for the fingers. """
    keypoints = [0.0 for _ in range(21)] # init empty list

    # fill keypoints which are regressed
    mapping = {0: 0, #Wrist
               1: 5, 2: 6, 3: 7, #Index
               4: 9, 5: 10, 6: 11, #Middle
               7: 17, 8: 18, 9: 19, # Pinky
               10: 13, 11: 14, 12: 15, # Ring
               13: 1, 14: 2, 15: 3} # Thumb

    for manoId, myId in mapping.items():
        keypoints[myId] = keypoints_regressed[manoId, :]

    # get other keypoints from mesh
    for myId, meshId in kpId2vertices.items():
        keypoints[myId] = ch.mean(mesh_vertices[meshId, :], 0)

    keypoints = ch.vstack(keypoints)

    return keypoints

