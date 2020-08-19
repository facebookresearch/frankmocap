import cv2
import numpy as np
import os.path as osp
import sys
import time
import ry_utils
import os


def load_checkpoints(checkpoint_dir):
    all_checkpoints = list()
    # for subdir, dirs, files in os.walk(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".pth"):
            all_checkpoints.append(osp.join(checkpoint_dir, file))
    return sorted(all_checkpoints)


def pad_and_resize(img, final_size=224):
    height, width = img.shape[:2]
    if height > width:
        ratio = final_size / height
        new_height = final_size
        new_width = int(ratio * width)
    else:
        ratio = final_size / width
        new_width = final_size
        new_height = int(ratio * height)
    new_img = np.zeros((final_size, final_size, 3), dtype=np.uint8)
    new_img[:new_height, :new_width, :] = cv2.resize(img, (new_width, new_height))
    return new_img


def build_dirs(pred_results, data_list, res_dir):
    for result in pred_results:
        img_name = data_list[result['data_idx']]['image_name']
        if img_name.find('/')>0:
            subdir = '/'.join(img_name.split('/')[:-1])
            res_subdir = osp.join(res_dir, subdir)
            ry_utils.build_dir(res_subdir)


def update_opt_from_train(train_file, opt):
    # read train file
    with open(train_file, "r") as in_f:
        all_lines = in_f.readlines()

    # update top_joints_finger_type    
    updated = False
    for line in all_lines:
        if line.find("top_joints_type=")>=0:
            value = line.strip().replace("top_joints_type=", "").strip()
            opt.top_finger_joints_type = value
            updated = True
    assert updated, "Initialize top finger joints type failed"
    return opt


def get_evaluate_type(test_dataset):
    if test_dataset.has_mano_anno:
        eval_type = 'pve'
    elif test_dataset.has_joints_3d:
        eval_type = 'auc_3d'
    elif test_dataset.evaluate_joints_2d:
        eval_type = 'auc_2d'
    else:
        assert test_dataset.evaluate_demo_video
        eval_type = 'seq_jitter'
    return eval_type



class Timer(object):
    def __init__(self, num_batch):
        self.start=time.time()
        self.num_batch = num_batch
    
    def click(self, batch_id):
        start, num_batch = self.start, self.num_batch
        end = time.time()
        cost_time = (end-start)/60
        speed = (batch_id+1)/cost_time
        res_time = (num_batch-(batch_id+1))/speed
        print("we have process {0}/{1}, it takes {2:.3f} mins, remain needs {3:.3f} mins".format(
            (batch_id+1), num_batch, cost_time, res_time))
        sys.stdout.flush()


class ResultStat(object):
    def __init__(self, results_info, eval_type):
        # save all results
        self.all_results = dict()
        self.best_results = dict()
        self.get_best_results = dict()
        for name, result_type, scale_ratio in results_info:
            assert result_type in ['large', 'small']
            self.all_results[name] = (result_type, scale_ratio, list())
            self.best_results[name] = None
            self.get_best_results[name] = False
        # evaluate which metric 
        self.eval_type = eval_type
    

    def update(self, name, epoch, value):
        # add to all results
        self.all_results[name][2].append((epoch, value))

        result_type = self.all_results[name][0]
        if result_type == 'large':
            if self.best_results[name] is None or value > self.best_results[name][0]:
                self.best_results[name] = (value, epoch)
                self.get_best_results[name] = True
            else:
                self.get_best_results[name] = False
        else:
            if self.best_results[name] is None or value < self.best_results[name][0]:
                self.best_results[name] = (value, epoch)
                self.get_best_results[name] = True
            else:
                self.get_best_results[name] = False
    

    def _get_valid_metric(self):
        if self.eval_type == 'pve':
            return ['pve', 'mpjpe_mano']
        elif self.eval_type == 'auc_3d':
            return ['mpjpe_3d', 'auc_3d']
        elif self.eval_type == 'auc_2d':
            return ['auc_2d']
        else:
            assert self.eval_type == 'seq_jitter'
            return ['seq_jitter',]
    

    def print_current_result(self, epoch):
        valid_metrics = self._get_valid_metric()
        print("Test of {} complete".format(epoch))
        print_content = ""
        for name in valid_metrics:
            result_type, scale_ratio, results = self.all_results[name]
            print_content += f"{name} : {results[-1][1]*scale_ratio:.4f}\t"
        print(print_content.strip())


    def print_best_results(self):
        valid_metrics = self._get_valid_metric()
        record_content = ""
        for name in valid_metrics:
            scale_ratio = self.all_results[name][1]
            best_result, best_epoch = self.best_results[name]
            best_result *= scale_ratio

            print("\n=============Best Result===============")
            print("{0}:{1:.4f}, model:{2}".format(name, best_result, best_epoch))
            record_content += "{0:.3f} ({1} epoch) / ".format(best_result, best_epoch)
        record_content = record_content[:-3] # remove the last " / "
        print(record_content)
    

    def achieve_better(self):
        return self.get_best_results[self.eval_type]