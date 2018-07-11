#!/bin/bash

cd ~/halo_cnn

#interact -p GPU-shared --gres=gpu:p100:1 -t 02:00:00
#interact --ntasks-per-node=27 -t 02:00:00

printf "\nRunning ML..."
module load keras/2.0.6_anaconda

source activate $KERAS_ENV

python ./scripts/halo_cnn3d_rcart/halo_cnn3d_rcart_ml.py

source deactivate

module unload keras/2.0.6_anaconda
