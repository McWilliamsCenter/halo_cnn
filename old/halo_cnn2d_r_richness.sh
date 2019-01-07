#!/bin/bash

cd ~/halo_cnn

#interact -p GPU-shared --gres=gpu:p100:1 -t 02:00:00
#interact --ntasks-per-node=27 -t 02:00:00

printf "Generating richness data..."

module load AI/anaconda3-5.1.0_gpu
source activate $AI_ENV

python ./scripts/halo_cnn2d_r/halo_cnn2d_r_rich_1.py


printf "\nCalculating richness dependance..."

python ./scripts/halo_cnn2d_r/halo_cnn2d_r_rich_2.py


