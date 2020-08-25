import os, sys, shutil
import os.path as osp

def main():
    for subdir, dirs, files in os.walk('log/test_logs'):
        for file in files:
            if file.endswith('.log'):
                dataset_name = file[:-4].split('_')[-1]
                if dataset_name != 'wild':
                    log_file = osp.join(subdir, file)
                    with open(log_file, 'r') as in_f:
                        lines = in_f.readlines()
                        print(dataset_name)
                        for line in lines[-3:]:
                            print(line.strip())
                        print("=================")

if __name__ == '__main__':
    main()