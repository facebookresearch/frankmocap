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
# Code Developed by: Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
# 2018.01.02

import os, glob
import numpy as np


def expid2model(expr_dir):
    from configer import Configer

    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)

    best_model_fname = sorted(glob.glob(os.path.join(expr_dir, 'snapshots', '*.pt')), key=os.path.getmtime)[-1]
    try_num = os.path.basename(best_model_fname).split('_')[0]

    print(('Found Trained Model: %s' % best_model_fname))

    default_ps_fname = glob.glob(os.path.join(expr_dir,'*.ini'))[0]
    if not os.path.exists(
        default_ps_fname): raise ValueError('Could not find the appropriate vposer_settings: %s' % default_ps_fname)
    ps = Configer(default_ps_fname=default_ps_fname, work_dir = expr_dir, best_model_fname=best_model_fname)

    return ps, best_model_fname

def load_vposer(expr_dir, vp_model='snapshot'):
    '''

    :param expr_dir:
    :param vp_model: either 'snapshot' to use the experiment folder's code or a VPoser imported module, e.g.
    from human_body_prior.train.vposer_smpl import VPoser, then pass VPoser to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    '''
    import importlib
    import os
    import torch

    ps, trained_model_fname = expid2model(expr_dir)
    if vp_model == 'snapshot':

        vposer_path = sorted(glob.glob(os.path.join(expr_dir, 'vposer_*.py')), key=os.path.getmtime)[-1]

        spec = importlib.util.spec_from_file_location('VPoser', vposer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        vposer_pt = getattr(module, 'VPoser')(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)
    else:
        vposer_pt = vp_model(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)

    vposer_pt.load_state_dict(torch.load(trained_model_fname, map_location='cpu'))
    vposer_pt.eval()

    return vposer_pt, ps


def extract_weights_asnumpy(exp_id, vp_model= False):
    from human_body_prior.tools.omni_tools import makepath
    from human_body_prior.tools.omni_tools import copy2cpu as c2c

    vposer_pt, vposer_ps = load_vposer(exp_id, vp_model=vp_model)

    save_wt_dir = makepath(os.path.join(vposer_ps.work_dir, 'weights_npy'))

    weights = {}
    for var_name, var in vposer_pt.named_parameters():
        weights[var_name] = c2c(var)
    np.savez(os.path.join(save_wt_dir,'vposerWeights.npz'), **weights)

    print(('Dumped weights as numpy arrays to %s'%save_wt_dir))
    return vposer_ps, weights

if __name__ == '__main__':
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    expr_dir = '/ps/project/humanbodyprior/VPoser/smpl/pytorch/0020_06_amass'
    from human_body_prior.train.vposer_smpl import VPoser
    vposer_pt, ps = load_vposer(expr_dir, vp_model='snapshot')
    pose = c2c(vposer_pt.sample_poses(10))
    print(pose.shape)