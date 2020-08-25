import os, sys, shutil
import os.path as osp
import cv2

def convert_format(in_dir, src_format, dst_format):
    for subdir, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith(src_format):
                img_path = osp.join(osp.join(subdir, file))
                img = cv2.imread(img_path)
                new_path = img_path.replace(src_format, dst_format)
                cv2.imwrite(new_path, img)
                os.remove(img_path)
                print(new_path)

def main():
    in_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_shared_02/mtc_output"
    src_format = "jpg"
    dst_format = "png"
    convert_format(in_dir, src_format, dst_format)

if __name__ == '__main__':
    main()