import os, sys, shutil
import os.path as osp
import cv2
import mocap_utils.general_utils as g_utils


def setup_render_out(render_out_dir):
    if render_out_dir is not None:
        g_utils.build_dir(render_out_dir)
        outputFileName = 'scene_%08d.jpg' # Hardcoded in glViewer.py

        overlaidImageFolder= osp.join(render_out_dir, 'overlaid')
        g_utils.build_dir(overlaidImageFolder)

        sideImageFolder= osp.join(render_out_dir, 'side')
        g_utils.build_dir(sideImageFolder)

        mergedImageFolder= osp.join(render_out_dir, 'merged')
        g_utils.build_dir(mergedImageFolder)

        g_renderDir= osp.join(render_out_dir, 'render')
        g_utils.build_dir(g_renderDir)

        res_subdirs = \
            [overlaidImageFolder, sideImageFolder, mergedImageFolder, g_renderDir]
        return res_subdirs
    
    else:
        return None


def __get_video_path(args):
    if args.video_type == 'webcam':
        video_path = 0
    elif args.video_type == 'url':
        if args.download:
            os.makedirs("./webvideos",exist_ok=True)
            downloadPath ="./webvideos/{0}.mp4".format(osp.basename(args.url))
            cmd_download = "youtube-dl -f best {0} -o {1}".format(args.url,downloadPath)
            print(">> Downloading: {}".format(args.url))
            print(">> {}".format(cmd_download))
            os.system(cmd_download)
            video_path = downloadPath
        else:
            try:
                import pafy
                url = args.url #'https://www.youtube.com/watch?v=c5nhWy7Zoxg'
                vPafy = pafy.new(url)
                play = vPafy.getbest(preftype="webm")
                video_path = url
            except:
                video_path = args.url
    else:
        video_path = args.video_path
    return video_path


def setup_input(args):
    if args.input_type == "video":
        video_path = __get_video_path(args)
        print("video_path", video_path)
        sys.exit(0)
        image_list =[]
        cap =None
        if osp.isdir(video_path):       #if video_path is a dir, load all videos
            image_list = sorted(os.listdir(video_path))
            image_list = [osp.join(video_path,f) for f in image_list]
        else:
            cap = cv2.VideoCapture(video_path)
            if osp.exists(video_path):
                print("valid")
            if cap.isOpened()==False:
                print(f"Failed in opening video: {video_path}")
                assert False
    else:
        img_exts = ('jpg', 'png', 'jpeg', 'bmp')
        image_list = g_utils.get_all_files(args.image_path, img_exts, "full") 
        return image_list, None