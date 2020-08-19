strategy_params = dict(
    # copy and paste
    copy_and_paste = dict(
        memory_size = 5, # considering memory_size of previous frames
        update_thresh = 0.05, # when openpose score is lower than thresh, copy and paste from previous samples
        select_thresh = 0.2, # samples with openpose thresh higher than this can be used to replace post frames
    ),

    # replace bad wrist rotation (from hand model) to fairmocap
    update_wrist = dict(
        threshold = 500,
    ),

    # average the adjacent frames
    average_frame = dict(
        win_size = 5,
    )
)