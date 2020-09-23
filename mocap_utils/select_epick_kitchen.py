# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
sys.path.append('mocap_utils')
from general_utils import get_all_files, renew_dir


def main():
    in_dir = 'samples/image/body/ego_centric/epic_kitchen'
    out_dir = 'samples/image/body/ego_centric/epic_kitchen_selected'
    renew_dir(out_dir)

    all_imgs = get_all_files(in_dir, '.jpg', 'relative')
    for img_id, img_name in enumerate(all_imgs):
        if img_id>=600 and img_id<900:
            in_img_path = osp.join(in_dir, img_name)
            out_img_path = osp.join(out_dir, img_name)
            shutil.copy2(in_img_path, out_img_path)


if __name__ == '__main__':
    main()