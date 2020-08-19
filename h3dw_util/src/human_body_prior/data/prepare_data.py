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

import os
import numpy as np
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.tools.omni_tools import euler2em, em2euler
from human_body_prior.tools.omni_tools import copy2cpu as c2c

import shutil, sys
from torch.utils.data import Dataset
import glob
from datetime import datetime
import torch

def remove_Zrot(pose):
    noZ = em2euler(pose[:3].copy())
    noZ[2] = 0
    pose[:3] = euler2em(noZ).copy()
    return pose

def dump_amass2pytroch(datasets, amass_dir, out_dir, split_name, logger = None, rnd_seed = 100):
    '''
    Select random number of frames from central 80 percent of each mocap sequence

    :param datasets:
    :param amass_dir:
    :param out_dir:
    :param split_name:
    :param logger
    :param rnd_seed:
    :return:
    '''
    import glob
    from tqdm import tqdm

    assert split_name in ['train', 'vald', 'test']
    np.random.seed(rnd_seed)

    makepath(out_dir, isfile=False)

    if logger is None:
        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        logger = log2file(os.path.join(out_dir, '%s.log' % (log_name)))
        logger('Creating pytorch dataset at %s' % out_dir)

    if split_name in ['vald', 'test']:
        keep_rate = 0.3  # this should be fixed for vald and test datasets
    elif split_name == 'train':
        keep_rate = 0.3  # 30 percent, which would give you around 3.5 M training data points

    data_pose = []
    data_betas = []
    data_gender = []
    data_trans = []
    data_markers = []

    for ds_name in datasets:
        npz_fnames = glob.glob(os.path.join(amass_dir, ds_name, '*/*_poses.npz'))
        logger('randomly selecting data points from %s.' % (ds_name))
        for npz_fname in tqdm(npz_fnames):
            cdata = np.load(npz_fname)
            N = len(cdata['poses'])

            # skip first and last frames to avoid initial standard poses, e.g. T pose
            cdata_ids = np.random.choice(list(range(int(0.1*N), int(0.9*N),1)), int(keep_rate*0.8*N), replace=False)
            if len(cdata_ids)<1: continue

            data_pose.extend(cdata['poses'][cdata_ids].astype(np.float32))
            data_trans.extend(cdata['trans'][cdata_ids].astype(np.float32))
            data_betas.extend(np.repeat(cdata['betas'][np.newaxis].astype(np.float32), repeats=len(cdata_ids), axis=0))
            data_gender.extend([{'male':-1, 'neutral':0, 'female':1}[str(cdata['gender'].astype(np.str))] for _ in cdata_ids])
            if split_name == 'test':
                data_markers.extend(np.repeat(cdata['betas'][np.newaxis].astype(np.float32), repeats=len(cdata_ids), axis=0))

    outdir = makepath(os.path.join(out_dir, split_name))

    assert len(data_pose) != 0

    outpath = os.path.join(outdir, 'pose.pt')
    torch.save(torch.tensor(np.asarray(data_pose, np.float32)), outpath)

    outpath = os.path.join(outdir, 'betas.pt')
    torch.save(torch.tensor(np.asarray(data_betas, np.float32)), outpath)

    outpath = os.path.join(outdir, 'trans.pt')
    torch.save(torch.tensor(np.asarray(data_trans, np.float32)), outpath)

    outpath = os.path.join(outdir, 'gender.pt')
    torch.save(torch.tensor(np.asarray(data_gender, np.int32)), outpath)

    logger('Len. split %s %d' %(split_name, len(data_pose)))

class AMASS_Augment(Dataset):
    """Use this dataloader to do any augmentation task in parallel"""

    def __init__(self, dataset_dir, dtype=torch.float32):

        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname)

        self.dtype = dtype

    def __len__(self):
       return len(self.ds['trans'])

    def __getitem__(self, idx):
        return self.fetch_data(idx)

    def fetch_data(self, idx):
        sample = {k: self.ds[k][idx] for k in self.ds.keys()}
        from human_body_prior.train.vposer_smpl import VPoser
        sample['pose_matrot'] = VPoser.aa2matrot(sample['pose'].view([1,1,-1,3])).view(1,-1)

        return sample

def prepare_vposer_datasets(amass_splits, amass_dir, vposer_datadir, logger=None):

    if logger is None:
        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        logger = log2file(os.path.join(vposer_datadir, '%s.log' % (log_name)))
        logger('Creating pytorch dataset at %s' % vposer_datadir)

    stageI_outdir = os.path.join(vposer_datadir, 'stage_I')

    shutil.copy2(sys.argv[0], os.path.join(vposer_datadir, os.path.basename(sys.argv[0])))

    logger('Stage I: Fetch data from AMASS npz files')

    for split_name, datasets in amass_splits.items():
        if os.path.exists(os.path.join(stageI_outdir, split_name, 'pose.pt')): continue
        dump_amass2pytroch(datasets, amass_dir, stageI_outdir, split_name=split_name, logger=logger)

    logger('Stage II: augment data by noise and save into h5 files to be used in a cross framework scenario.')
    ## Writing to h5 files is also convinient since appending to files is possible
    from torch.utils.data import DataLoader
    import tables as pytables
    from tqdm import tqdm

    class AMASS_ROW(pytables.IsDescription):

        gender = pytables.Int16Col(1)  # 1-character String
        pose = pytables.Float32Col(52*3)  # float  (single-precision)
        pose_matrot = pytables.Float32Col(52*9)  # float  (single-precision)
        betas = pytables.Float32Col(16)  # float  (single-precision)
        trans = pytables.Float32Col(3)  # float  (single-precision)

    stageII_outdir = makepath(os.path.join(vposer_datadir, 'stage_II'))

    batch_size = 256
    max_num_epochs = 1  # how much augmentation we would get

    for split_name in amass_splits.keys():
        h5_outpath = os.path.join(stageII_outdir, '%s.h5' % split_name)
        if os.path.exists(h5_outpath): continue

        ds = AMASS_Augment(dataset_dir=os.path.join(stageI_outdir, split_name))
        logger('%s has %d data points!' % (split_name, len(ds)))
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=32, drop_last=False)
        with pytables.open_file(h5_outpath, mode="w") as h5file:
            table = h5file.create_table('/', 'data', AMASS_ROW)

            for epoch_num in range(max_num_epochs):
                for bId, bData in tqdm(enumerate(dataloader)):
                    for i in range(len(bData['trans'])):
                        for k in bData.keys():
                            table.row[k] = c2c(bData[k][i])
                        table.row.append()
                    table.flush()

    logger('Stage III: dump every thing as a final thing to pt files')
    # we would like to use pt files because their interface could run in multiple threads
    stageIII_outdir = makepath(os.path.join(vposer_datadir, 'stage_III'))

    for split_name in amass_splits.keys():
        h5_filepath = os.path.join(stageII_outdir, '%s.h5' % split_name)
        if not os.path.exists(h5_filepath) : continue

        with pytables.open_file(h5_filepath, mode="r") as h5file:
            data = h5file.get_node('/data')
            data_dict = {k:[] for k in data.colnames}
            for id in range(len(data)):
                cdata = data[id]
                for k in data_dict.keys():
                    data_dict[k].append(cdata[k])

        for k,v in data_dict.items():
            outfname = makepath(os.path.join(stageIII_outdir, split_name, '%s.pt' % k), isfile=True)
            if os.path.exists(outfname): continue
            torch.save(torch.from_numpy(np.asarray(v)), outfname)

    logger('Dumped final pytorch dataset at %s' % stageIII_outdir)