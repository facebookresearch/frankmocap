import os, sys, shutil
import os.path as osp
import ry_utils
import parallel_io as pio


def main():
    in_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/models/pretrained_weights"
    all_files = ry_utils.get_all_files(in_dir, ".pth", "relative")
    for file in all_files:
        full_path = osp.join(in_dir, file)
        print(file, ry_utils.md5sum(full_path))


if __name__ == '__main__':
    main()