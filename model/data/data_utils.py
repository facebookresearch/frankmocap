import os
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os.path as osp
import parallel_io as pio
from scipy.io import loadmat

def _load_data_from_dir(img_dir):
    data_list = list()
    for subdir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith( ("jpg", "jpeg", "png") ):
                img_name = osp.join(subdir, file).replace(img_dir, "")
                if img_name[0] == "/": # remove the potential "/" in the head of path
                    img_name = img_name[1:]
                single_data = dict(
                    image_name = img_name
                )
                data_list.append(single_data)
    assert len(data_list)>0, "Given Directory contains no image."
    return data_list


def load_annotation(data_root, anno_path, use_augment=True):
    anno_path_full = osp.join(data_root, anno_path)
    if osp.isdir(anno_path_full):
        all_data = _load_data_from_dir(anno_path_full)
    else:
        all_data = pio.load_pkl_single(anno_path_full)
    
    if isinstance(all_data, list):
        data_list = all_data
    else:
        raise ValueError("Unsupported data type")
    return data_list


def load_blur_kernel(blur_kernel_dir):
    kernels = list()
    for file in os.listdir(blur_kernel_dir):
        if file.endswith(".mat"):
            kernel = loadmat(osp.join(blur_kernel_dir, file))['PSFs'][0][0]
            kernels.append(kernel)
    return kernels