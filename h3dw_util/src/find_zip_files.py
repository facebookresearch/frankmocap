import os, sys, shutil
import os.path as osp


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    total_file_size = 0.0
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".zip"):
                if file.find("frl_full")<0:
                    file_path = osp.join(subdir, file)
                    file_size = osp.getsize(file_path) / 1024.0 / 1024.0 / 1024.0
                    total_file_size += file_size
                    print(file_path, file_size)
                    os.remove(file_path)
    print(total_file_size)
    

if __name__ == '__main__':
    main()