'''
Copyright 2017 Javier Romero, Dimitrios Tzionas, Michael J Black and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the MANO/SMPL+H Model license here http://mano.is.tue.mpg.de/license

More information about MANO/SMPL+H is available at http://mano.is.tue.mpg.de.
For comments or questions, please email us at: mano@tue.mpg.de


About this file:
================
This file defines a wrapper for the loading functions of the MANO model. 

Modules included:
- load_model:
  loads the MANO model from a given file location (i.e. a .pkl file location), 
  or a dictionary object.

'''

def ready_arguments(fname_or_dict, posekey4vposed='pose'):
    import numpy as np
    import pickle
    import chumpy as ch
    from chumpy.ch import MatVecMult
    from utils.freihand_utils.mano_core.posemapper import posemap

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict))
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1]*3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    assert(posekey4vposed in dd)
    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'].dot(dd['betas'])+dd['v_template']
        v_shaped = dd['v_shaped']
        J_tmpx = MatVecMult(dd['J_regressor'], v_shaped[:, 0])
        J_tmpy = MatVecMult(dd['J_regressor'], v_shaped[:, 1])
        J_tmpz = MatVecMult(dd['J_regressor'], v_shaped[:, 2])
        dd['J'] = ch.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'].dot(posemap(dd['bs_type'])(dd[posekey4vposed]))
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'].dot(posemap(dd['bs_type'])(dd[posekey4vposed]))

    return dd


def load_model(fname_or_dict, ncomps=6, flat_hand_mean=False, v_template=None, use_pca=True):
    ''' This model loads the fully articulable HAND SMPL model,
    and replaces the pose DOFS by ncomps from PCA'''

    # ("use_pca", use_pca) # False
    # print("flat_hand_mean", flat_hand_mean) # False

    from .verts import verts_core
    import numpy as np
    import chumpy as ch
    import pickle
    import scipy.sparse as sp
    np.random.seed(1)

    if not isinstance(fname_or_dict, dict):
        smpl_data = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
    else:
        smpl_data = fname_or_dict

    rot = 3  # for global orientation!!!

    if use_pca:
        hands_components = smpl_data[hands_components]  # PCA components
    else:
        hands_components = np.eye(45)  # directly modify 15x3 articulation angles
    
    # print("hands_components", hands_components)
    hands_mean       = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data['hands_mean']
    hands_coeffs     = smpl_data['hands_coeffs'][:, :ncomps]

    # print('FreiHand: hands_mean', hands_mean)
    # print("FreiHand: hands_coeffs", hands_coeffs)
    # print("FreiHand: hands_mean", hands_mean.shape)

    selected_components = np.vstack((hands_components[:ncomps]))
    hands_mean = hands_mean.copy()

    pose_coeffs = ch.zeros(rot + selected_components.shape[0])
    full_hand_pose = pose_coeffs[rot:(rot+ncomps)].dot(selected_components)


    smpl_data['fullpose'] = ch.concatenate((pose_coeffs[:rot], hands_mean + full_hand_pose))

    '''
    print(selected_components)
    print("hands_mean", hands_mean.shape)
    print("hands_components", hands_components.shape)
    print(hands_components - selected_components)
    print("selected_components", selected_components.shape)
    '''

    smpl_data['pose'] = pose_coeffs
    # smpl_data['fullpose'] = ch.array(smpl_data['fullpose'].r)

    Jreg = smpl_data['J_regressor']
    if not sp.issparse(Jreg):
        smpl_data['J_regressor'] = (sp.csc_matrix((Jreg.data, (Jreg.row, Jreg.col)), shape=Jreg.shape))

    # slightly modify ready_arguments to make sure that it uses the fullpose 
    # (which will NOT be pose) for the computation of posedirs
    dd = ready_arguments(smpl_data, posekey4vposed='fullpose')

    # create the smpl formula with the fullpose,
    # but expose the PCA coefficients as smpl.pose for compatibility
    args = {
        'pose': dd['fullpose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': dd['bs_style'],
    }

    result_previous, meta = verts_core(**args)
    result = result_previous + dd['trans'].reshape((1, 3))
    result.no_translation = result_previous

    if meta is not None:
        for field in ['Jtr', 'A', 'A_global', 'A_weighted']:
            if(hasattr(meta, field)):
                setattr(result, field, getattr(meta, field))

    if hasattr(result, 'Jtr'):
        result.J_transformed = result.Jtr + dd['trans'].reshape((1, 3))

    for k, v in dd.items():
        setattr(result, k, v)

    if v_template is not None:
        result.v_template[:] = v_template
    result.dd = dd
    return result

if __name__ == '__main__':
    m = load_model()
    m.J_transformed
    print('FINITO')
