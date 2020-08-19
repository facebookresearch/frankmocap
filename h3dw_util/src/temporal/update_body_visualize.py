import os, sys, shutil
import os.path as osp
import ry_utils


def load_imgs(img_dir):
    all_imgs = dict()
    for subdir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(".jpg"):
                img_path = osp.join(subdir, file)
                img_name = '/'.join(img_path.split('/')[-2:])
                all_imgs[img_name] = img_path
    return all_imgs


def main():
    origin_img_dir = f"visualization/temporal/render_body/origin/hand_wrist_rot/"
    all_imgs_origin = load_imgs(origin_img_dir)

    update_img_dir = f"visualization/temporal/render_body/update_wrist/updated/"
    all_imgs_update = load_imgs(update_img_dir)

    res_dir = f"visualization/temporal/render_body/update_wrist/full"
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