#!/bin/bash

cd ~/halo_cnn

#interact -p GPU-shared --gres=gpu:p100:1 -t 02:00:00
#interact --ntasks-per-node=27 -t 02:00:00

printf "Generating data..."

module load python3/intel_3.6.3

python ./scripts/halo_cnn2d_r/halo_cnn2d_r_data.py

module unload python3/intel_3.6.3
printf "Data generated.\n"

