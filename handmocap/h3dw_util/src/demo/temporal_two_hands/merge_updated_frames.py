import os, sys, shutil
import os.path as osp
import ry_utils


def load_imgs(img_dir):
    all_imgs = dict()
    all_files = ry_utils.get_all_files(img_dir, ".jpg", "relative")
    for file in all_files:
        all_imgs[file] = osp.join(img_dir, file)
    return all_imgs


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube/"

    # origin_img_dir = osp.join(root_dir, "augment_bbox/prediction/h3dw/origin_frame")
    origin_img_dir = osp.join(root_dir, "temporal_refine/origin_frame")
    all_imgs_origin = load_imgs(origin_img_dir)

    update_img_dir = osp.join(root_dir, "temporal_refine/update_frame/copy_and_paste")
    all_imgs_update = load_imgs(update_img_dir)

    res_dir = osp.join(root_dir, "temporal_refine/merge_frame/copy_and_paste")
    ry_utils.renew_dir(res_dir)

    for img_name in all_imgs_origin:
        if img_name in all_imgs_update:
            img_path = all_imgs_update[img_name]
        else:
            img_path = all_imgs_origin[img_name]
        res_img_path = osp.join(res_dir, img_name)
        ry_utils.make_subdir(res_img_path)
        shutil.copy2(img_path, res_img_path)


if __name__ == '__main__':
    main()