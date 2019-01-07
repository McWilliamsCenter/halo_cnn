#!/bin/bash

cd ~/halo_cnn

#interact -p GPU-shared --gres=gpu:p100:1 -t 02:00:00
#interact --ntasks-per-node=27 -t 02:00:00

printf "Running M(sigma) regression..."

module load anaconda3

source activate jupy

python ./scripts/halo_cnn2d_r/halo_cnn2d_r_regr.py

source deactivate

module unload anaconda3

