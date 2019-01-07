#!/bin/bash

cd ~/halo_cnn

#interact -p GPU-shared --gres=gpu:p100:1 -t 02:00:00
#interact --ntasks-per-node=27 -t 02:00:00

printf "Generating data..."

module load anaconda5
source activate jupy

python ./scripts/halo_cnn1d_r/halo_cnn1d_r_data.py

source deactivate
module unload anaconda5
printf "Data generated.\n"



printf "\nRunning ML..."
module load keras/2.0.6_anaconda

source activate $KERAS_ENV

python ./scripts/halo_cnn1d_r/halo_cnn1d_r_ml.py

source deactivate

module unload keras/2.0.6_anaconda


printf "Running M(sigma) regression..."

module load anaconda5

source activate jupy

python ./scripts/halo_cnn1d_r/halo_cnn1d_r_regr.py

source deactivate

module unload anaconda5


printf "\nPlotting..."
module load python3/intel_3.6.3

python ./scripts/halo_cnn1d_r/halo_cnn1d_r_plot.py

module unload python3/intel_3.6.3
