import os, sys, shutil
import os.path as osp
import ry_utils


def main():
    in_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_shared_02/mtc_output"

    all_txt_files = ry_utils.get_all_files(in_dir, ".txt", "full")
    for file in all_txt_files:
        os.remove(file)

    all_files = ry_utils.get_all_files(in_dir, ".png", "full")
    for file in all_files:
        record = file.split('/')
        img_name = record[-1]
        seq_name = record[-2]
        new_img_name = seq_name + '_' + img_name
        new_img_path = file.replace(img_name, new_img_name)
        shutil.copy2(file, new_img_path)
        os.remove(file)


if __name__ == '__main__':
    main()