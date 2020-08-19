import os, sys, shutil
import os.path as osp
import ry_utils

def main():
    frame_dir = "frame_origin"

    for subdir, dirs, files in os.walk(frame_dir):
        for file_name in files:
            if file_name.endswith(".png"):
                record = file_name.split('.')
                file_id = int(record[0].split('_')[-1])
                prefix = '_'.join(record[0].split('_')[:-1])
                extension = record[1]
                new_file_name = f"{prefix}_{file_id:05d}.{extension}"
                old_path = osp.join(subdir, file_name)
                new_path = osp.join(subdir, new_file_name)
                shutil.move(old_path, new_path)

if __name__ == '__main__':
    main()