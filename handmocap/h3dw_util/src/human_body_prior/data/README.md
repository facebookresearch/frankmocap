# Preparing VPoser Training Dataset
The Human Body Prior, VPoser, presented here is trained on  [AMASS](https://amass.is.tue.mpg.de/) dataset. 
AMASS is a large collection of human marker based optical mocap data as [SMPL](http://smpl.is.tue.mpg.de/) body model parameters.
VPoser code here is implemented in [PyTorch](https://pytorch.org/), therefore, the data preparation code, 
turns AMASS data into pytorch readable *.pt* files in three stages:

***Stage I*** turns the AMASS numpy *.npz* files into PyTorch *.pt* files. 
For this, first you would need to download body parameters from the AMASS webpage: https://amass.is.tue.mpg.de/dataset.
Then you have to select subsets of AMASS to be used for each data splits, e.g. train/validation/test. 
Here we follow the recommended data splits of AMASS, that is:

```python
amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BML', 'EKUT', 'TCD_handMocap', 'ACCAD']
}
amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))
```

During this stage, we also subsample the original data, so that we only take every some frames of the original mocap
to be included in the final data files. 
 
***Stage II*** turns the AMASS pytorch files into HDF5, *h5* files and along the process augments the data with extra fields or noise. 
Using pytorch in the middle stage helps to parallelize augmentation tasks. 
Furthermore, we use HDF5 files for the middle stage so that they can be used in other deep learning frameworks as well.

***Stage III*** again converts the augmented HDF5 files into final pytorch files that should be provided to the current VPoser training script.

During the process, the data preparation code can dump a log file to make it possible to track how data for different
experiments has been produced.

Below is a full python script example to prepare a VPoser training data:

```python
import os
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.data.prepare_data import prepare_vposer_datasets

expr_code = 'SOME_UNIQUE_ID'

amass_dir = 'THE_PATH_TO_AMASS_NPZ_FILES'

vposer_datadir = makepath('OUTPUT_DATA_PATH/%s' % (expr_code))

logger = log2file(os.path.join(vposer_datadir, '%s.log' % (expr_code)))
logger('[%s] Preparing data for training VPoser.'%expr_code)

amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BML', 'EKUT', 'TCD_handMocap', 'ACCAD']
}
amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))

prepare_vposer_datasets(amass_splits, amass_dir, vposer_datadir, logger=logger)
```

## Note
If you consider training your own VPoser for your research using AMASS dataset, then please follow its respective citation guideline. 