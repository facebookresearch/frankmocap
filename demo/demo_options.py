import argparse

class DemoOptions():

    def __init__(self):
        parser = argparse.ArgumentParser()
        
        default_checkpoint = "../data/weights/hand_module/checkpoints_best/pose_shape_best.pth"
        parser.add_argument('--checkpoint', required=False, default=default_checkpoint, help='Path to pretrained checkpoint')

        parser.add_argument('--crop_type', type=str, default='image', choices=['hand_crop', 'no_crop'],
            help = """ 'hand_crop' means the hand are central cropped in input. (left hand should be flipped to right). 
                        'no_crop' means perform hand detection is required to obtain hand bbox""")

        parser.add_argument('--input_type', type=str, default='image', choices=['image', 'video'],
            help = 'The type of input. It could be single image, sequence input (video)')
        parser.add_argument('--video_type', type=str, default='frame', choices=['frame', 'video', 'url'], 
            help = "If the input type is video, the type could be frame, video (eg.*.mp4), or url (youtube url)")
        parser.add_argument('--video_path', type=str, default=None, help="""Path of video or first image in a folder
                            (example: (path)/out%%1d.jpg - %%1d will be automatically replaced by the number of the image)
                            . Can also be used to load a single image (example: (path)/out1.jpg).""")
        parser.add_argument('--video_url', type=str, default=None, help='URL of YouTube video, or image.')
        parser.add_argument('--download', '-d', action='store_true', help='Download YouTube video first (in webvideo folder), and process it')
        parser.add_argument('--image_path', type=str, default=None, help='Path of directories that stores the input image')

        parser.add_argument("--renderer_type", type=str, default="opendr", choices=['opendr', 'opengl_gui', 'opengl_no_gui'], 
            help="Type of renderer used to render the predicted mesh")
        parser.add_argument('--render_out_dir', type=str, default=None, help='Folder of output images.')
        parser.add_argument('--pklout', action='store_true', help='Export mocap output as pkl file')
        parser.add_argument('--bUseSMPLX', action='store_true', help='use SMPLX instead of SMPL. You should use a model trained with SMPL-X')
        parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
        parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
        parser.add_argument('--noVis', action='store_true', help='Do not visualize output on the screen')
        parser.add_argument('--startFrame', type=int, default=0, help='given a sequence of frames, set the starting frame')
        parser.add_argument('--endFrame', type=int, default=-1, help='given a sequence of frames, set the last frame')
        parser.add_argument('--noVideoOut', action='store_true', help='Do not generate output video (ffmpeg)')
        parser.add_argument('--single', action='store_true', help='Reconstruct only one person in the scene with the biggest bbox')
        parser.add_argument('--skip', action='store_true', help='Skip there exist already processed outputs')

        self.parser = parser
    

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
