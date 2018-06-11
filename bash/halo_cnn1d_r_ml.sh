#!/bin/bash

cd ~/halo_cnn

#interact -p GPU-shared --gres=gpu:p100:1 -t 02:00:00
#interact --ntasks-per-node=27 -t 02:00:00

echo "Generating data..."
python ./scripts/halo_cnn1d_r_data.py
echo "Data generated."

module load keras/2.0.6_anaconda

source activate $KERAS_ENV

echo "running ml"
python ./scripts/halo_cnn1d_r_ml.py

source deactivate

module unload keras/2.0.6_anaconda

echo "plotting"
module load python3/intel_3.6.3

python ./scripts/halo_cnn1d_r_plot.py

module unload python3/intel_3.6.3
