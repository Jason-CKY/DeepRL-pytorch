# Deep RL policies on Pybullet Environments
This repo is a pytorch implementation of training a variational autoencoder (VAE). This is written to train a VAE for use in a RL environment, and contains code to generate images from random exploration of various RL environments from [pybullet](https://github.com/bulletphysics/bullet3) and [RLBench](https://github.com/stepjam/RLBench) for training.

## How to use
* clone this repo
* cd to DeepRL-pytorch/VAE
All scripts for VAE training and data generation are to be run in this directory

### Generating data from openai gym environment
```
python generate_data.py
usage: generate_data.py [-h] --env ENV --num_samples NUM_SAMPLES
                        [--max_ep_len MAX_EP_LEN] [--seed SEED] [--rlbench]
                        [--view {wrist_rgb,front_rgb,left_shoulder_rgb,right_shoulder_rgb}]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             environment_id
  --num_samples NUM_SAMPLES
                        specify number of image samples to generate
  --max_ep_len MAX_EP_LEN
                        Maximum length of an episode
  --seed SEED           seed number for reproducibility
  --rlbench             if true, use rlbench environment wrappers
  --view {wrist_rgb,front_rgb,left_shoulder_rgb,right_shoulder_rgb}
                        choose the type of camera view to generate image (only
                        for RLBench envs)
```

### Training VAE
```
python train_vae.py
usage: train_vae.py [-h] [--dir DIR] [--seed SEED] [--num_workers NUM_WORKERS]
                    [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--beta BETA]
                    [--lr LR] [--ngpu NGPU] [--save_freq SAVE_FREQ]
                    [--log_freq LOG_FREQ] [--save_dir SAVE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --dir DIR             path to image folders
  --seed SEED           seed number for reproducibility
  --num_workers NUM_WORKERS
                        number of workers for dataloaders
  --batch_size BATCH_SIZE
                        batch size
  --epochs EPOCHS       Number of epochs
  --beta BETA           Weighing value for KLD in B-VAE
  --lr LR               Learning Rate
  --ngpu NGPU           number of gpus to use
  --save_freq SAVE_FREQ
                        save weights every <x> iterations
  --log_freq LOG_FREQ   log losses every <x> iterations
  --save_dir SAVE_DIR   path to save weights and logs
```