CUDA_VISIBLE_DEVICES=4 python3 -m demo.demo_handmocap \
    --input_type image \
    --crop_type no_crop \
    --view_type ego_centric \
    --input_image_dir samples/image/body/selected \
    --render_out_dir samples/output/body/selected \
    --renderer_type opendr
