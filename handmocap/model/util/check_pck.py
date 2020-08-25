import os, sys, shutil
import os.path as osp
sys.path.append("util")
import metric_utils
import parallel_io as pio

def main():
    for file in os.listdir("evaluate_results"):
        if file.startswith("pck_pmhand") and file.endswith(".pkl"):
            file_path = osp.join("evaluate_results", file)
            all_pck = pio.load_pkl_single(file_path)
            auc = metric_utils.calc_auc_2d(all_pck)
            print(file, auc)

if __name__ == '__main__':
    main()