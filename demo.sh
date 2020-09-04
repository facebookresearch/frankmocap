CUDA_VISIBLE_DEVICES=4 python3 -m demo.demo_handmocap \
    --input_type image \
    --crop_type no_crop \
    --view_type third_view \
    --input_image_dir samples/image/body/third_view \
    --render_out_dir samples/output/body/third_view \
    --renderer_type opendr
