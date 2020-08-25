import ry_utils
import os.path as osp

def main():
    in_dir = "checkpoints_ablation"
    info_file = osp.join(in_dir, "readme.txt")
    print(info_file, osp.exists(info_file))
    with open(info_file, 'r') as in_f:
        for line in in_f:
            record = line.strip().split()
            file1 = record[0]
            file2 = osp.join(in_dir, record[1])
            md51 = ry_utils.md5sum(file1)
            md52 = ry_utils.md5sum(file2)
            print(file2, md51, md52)

if __name__ == '__main__':
    main()