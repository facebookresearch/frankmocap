import os, sys, shutil
import os.path as osp
import subprocess as sp
import ry_utils


def get_openpose_output(frame_dir, openpose_dir, res_dir):
    for file in os.listdir(frame_dir):
        file_path = osp.join(frame_dir, file)
        if osp.isdir(file_path):
            frame_subdir = file_path
            res_subdir = osp.join(res_dir, file)
            ry_utils.renew_dir(res_subdir)

            frame_subdir = osp.abspath(frame_subdir)
            res_subdir = osp.abspath(res_subdir)
            current_path = os.getcwd()
            os.chdir(openpose_dir)
            command = f"build/examples/openpose/openpose.bin --hand --image_dir {frame_subdir} --write_json {res_subdir} " + \
                f"--render_pose 0 --display 0 -model_pose BODY_25 --number_people_max 1"
            command = command.split()
            sp.run(command)
            os.chdir(current_path)
            print(f"{res_subdir}, completes")


def main():
    frame_dir = "frame"
    openpose_dir = "/home/rongyu/work/openpose"
    res_dir = "openpose_output"
    ry_utils.renew_dir(res_dir)

    get_openpose_output(frame_dir, openpose_dir, res_dir)

if __name__ == '__main__':
    main()