# Train VPoser from Scratch
You can train you own VPoser using the *vposer_smpl.py* code. First you would need to download the 
[AMASS](https://amass.is.tue.mpg.de/) dataset, then following the [data preparation tutorial](../data/README.md)
prepare the data for training. Afterwards, you can use the following code snippet to train your vposer:

```python
from human_body_prior.train.vposer_smpl import run_vposer_trainer
from configer import Configer

expr_code = 'SOME_UNIQUE_ID'
args = {
    'expr_code' : expr_code,
    'base_lr': 0.005,

    'dataset_dir': 'VPOSER_DATA_DIR_PRODUCED_BEFORE',
    'work_dir': 'BASE_WORKing_DIR/%s'%expr_code, # Later you will give this pass to vposer_loader to load the model
}
ps = Configer(default_ps_fname='./vposer_smpl_defaults.ini', **args) # This is the default configuration

# Make a message to describe the purpose of this experiment
expr_message = '\n[%s] %d H neurons, latentD=%d, batch_size=%d,  kl_coef = %.1e\n' \
               % (ps.expr_code, ps.num_neurons, ps.latentD, ps.batch_size, ps.kl_coef)
expr_message += '\n'
ps.expr_message = expr_message

run_vposer_trainer(ps)
``` 
The above code uses [Configer](https://github.com/nghorbani/configer) to handle configurations. 
It loads the default settings in *vposer_smpl_defaults.ini* and overloads it with your new args. 
You can also start the training from the command line:
```bash
python3 -m human_body_prior.train.vposer_smpl ./vposer_smpl_defaults.ini
```
The training code, will dump a log file along with tensorboard readable events file.