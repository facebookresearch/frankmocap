## Preparation
* Please put ry_utils.py and parallel_io.py in your Python search path. Latest version of two files can be found in tools/.
* Please use the SMPL-X modified by Yu.
* Please install package listed in doc/requirements.txt




## Model

### Train

#### Training Preparation
I use visdom for real-time visualization, so please start visdom first before training. To start visdom, please make a new tmux window and run
```
sh tools/start_visdom.sh
```
It will start visdom for port from 8097 to 8130  
To visualize the result on your local machine, please run the following command on your local machine  
```
python tools/connect_visdom.py $port_number
```


#### Training Command
To train model using one local GPU, please run   
```
sh script/train_single.sh
```

To train model using multi GPUs and FAIR cluster, please run  
```
sh script/train.sh
```

There are arguments in these training scripts, please refer to model/option/base_options.py and model/option/train_options.py. If they are confusing, please ask Yu with no hesitate.


### Evaluation

#### Weights
* model/checkpoints_best: Best models, to run demo data, please use
* model/checkpoints_good: Other good models. Please be aware that the best number of MPII-Hand is obtained with pose_shape_prob-0.7_1e-4_reg-1.0_epoch_150.pth in this dir
* model/checkpoints_ablation: Models for ablation study (MPII-Hand only)
* model/checkpoints_all: Combination of checkpoints_best & checkpoints_good & checkpoints_ablation
* model/checkpoints_no_shape_param: Best models trained without shape param
* model/checkpoints_stb: Models for STB dataset
* model/checkpoints_rhd: Models for RHD dataset


#### Evaluate Single Dataset
!!!!! Attention: This part only covers evaluate on other datasets except STB & RHD.

To evaluate sinlge dataset, please run 
```
sh script/test_single.sh
```
You can check model/log/test_logs for testing result  
By default, it will test on MPII-Hand dataset and test all weights stored in model/checkpoints.   
To test weights stored in other directory, please change the $test_checkpoint_dir in script/test_single.sh  
To test other dataset, please change the $test_dataset in script/test_single.sh  


#### Evaluate Multiple Dataset
!!!!! Attention: This part only covers evaluate on other datasets except STB & RHD.  
To evaluate multiple datasets simultaneously, please run
```
sh script/test_multi.sh
```
There will be more than one testing process and they will run in background. The other settings and log will be the same as evaluate single dataset.


#### Demo
To extract hand poses for any data, please run 
```
sh script/demo.sh
```
By running this script, it will generate prediction for all the weights stored in $test_dataset_dir and results will be stored in model/evaluate_results.  

Please change $demo_img_dir in script/demo.sh to any directories that stores cropped hand images. The code will recursively search the whole directory and test on any found jpg/png files. The original directory relationship will be keeped.  

One thing need to pay attention is that the path in $demo_img_dir is a relative path, it should be combined with $data_root to get the full path. This rule applies to all the path defined in train/test/demo script files.  

To visualize the predicted results, Please run. The results will be stored in model/evaluate_results/images
```
sh script/visualize_prediction.sh
```

### Models without shape params
Please checkout to no_shape_param branch, then train/val/demo will be the same.


### Evaluate STB & RHD
Please checkout to eval_stb_rhd branch, to evaluate STB, run

!!!!! Attention: This branch is tricky, please only use it for evaluating STB & RHD

To evaluate STB, run
```
sh script/test_stb.sh
```

To evaluate RHD, run
```
sh script/test_rhd.sh
```



## Temporal Refinement
Please refer to [docs/refinement.md](docs/refinement.md)


## Prepare Dataset and other
Please refer to [docs/prepare_dataset.md](docs/prepare_dataset.md)


# License
## License
[CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 
