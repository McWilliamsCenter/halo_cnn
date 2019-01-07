#!/bin/bash

cd ~/halo_cnn

#interact -p GPU-shared --gres=gpu:p100:1 -t 02:00:00
#interact --ntasks-per-node=27 -t 02:00:00


module load AI/anaconda3-5.1.0_gpu
source activate $AI_ENV


printf "\nCalculating richness dependance..."

python ./scripts/halo_cnn2d_r/halo_cnn2d_r_rich.py

source deactivate

module unload AI/anaconda3-5.1.0_gpu


