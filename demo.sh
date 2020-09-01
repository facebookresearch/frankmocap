CUDA_VISIBLE_DEVICES=4 python3 -m demo.demo_handmocap \
    --input_type image \
    --crop_type hand_crop \
    --image_path samples/image/hand_only \
    --render_out_dir render_result \
    --renderer_type opendr
