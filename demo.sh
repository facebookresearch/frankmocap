for view_type in third_view ego_centric
do
    CUDA_VISIBLE_DEVICES=4 python3 -m demo.demo_handmocap \
        --input_type image \
        --crop_type no_crop \
        --view_type $view_type \
        --input_image_dir samples/image/body/$view_type/selected \
        --render_out_dir samples/output/body/$view_type/selected \
        --renderer_type opendr
done

CUDA_VISIBLE_DEVICES=4 python3 -m demo.demo_handmocap \
    --input_type image \
    --crop_type hand_crop \
    --view_type $view_type \
    --input_image_dir samples/image/hand_only/selected \
    --render_out_dir samples/output/hand_only/selected \
    --renderer_type opendr
