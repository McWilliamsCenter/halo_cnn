#!/bin/bash

cd ~/halo_cnn

#interact -p GPU-shared --gres=gpu:p100:1 -t 02:00:00
#interact --ntasks-per-node=27 -t 02:00:00

printf "Generating data..."

module load python3/intel_3.6.3

python ./scripts/halo_cnn3d_rcyl/halo_cnn3d_rcyl_data.py

module unload python3/intel_3.6.3
printf "Data generated.\n"



printf "\nRunning ML..."
module load keras/2.0.6_anaconda

source activate $KERAS_ENV

python ./scripts/halo_cnn3d_rcyl/halo_cnn3d_rcyl_ml.py

source deactivate

module unload keras/2.0.6_anaconda



printf "\nPlotting..."
module load python3/intel_3.6.3

python ./scripts/halo_cnn3d_rcyl/halo_cnn3d_rcyl_plot.py

module unload python3/intel_3.6.3
