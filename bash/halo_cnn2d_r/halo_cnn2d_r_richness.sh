#!/bin/bash

cd ~/halo_cnn

#interact -p GPU-shared --gres=gpu:p100:1 -t 02:00:00
#interact --ntasks-per-node=27 -t 02:00:00

printf "Generating richness data..."

module load anaconda3
source activate jupy

# python ./scripts/halo_cnn2d_r/halo_cnn2d_r_rich_1.py

source deactivate
module unload anaconda3


printf "\nCalculating richness dependance..."
module load keras/2.0.6_anaconda

source activate $KERAS_ENV

python ./scripts/halo_cnn2d_r/halo_cnn2d_r_rich_2.py

source deactivate

module unload keras/2.0.6_anaconda
