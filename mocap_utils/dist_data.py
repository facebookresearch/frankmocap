import os, sys, shutil
import os.path as osp
import general_utils as gnu


def main():
    root_dir = "/mnt/SSD/rongyu/data/3D/frank_mocap/dataset/expose"
    in_dir = osp.join(root_dir, "image_crop/train")
    out_dir = osp.join(root_dir, "image_dist/train")
    gnu.renew_dir(out_dir)

    num_gpu = 7
    all_imgs = gnu.get_all_files(in_dir, (".jpg", ".png"), "name_only")
    num_data = len(all_imgs)
    num_each = num_data // num_gpu
    for i in range(num_gpu):
        start = i * num_each
        end = (i+1) * num_each if i<num_gpu-1 else num_data
        for j in range(start, end):
            out_subdir = osp.join(out_dir, str(i))
            img_name = all_imgs[j]
            in_path = osp.join(in_dir, img_name)
            out_path = osp.join(out_subdir, img_name)
            gnu.make_subdir(out_path)
            shutil.copy2(in_path, out_path)


if __name__ == '__main__':
    main()