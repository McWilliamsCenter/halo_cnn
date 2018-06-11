#!/bin/bash

cd ~/halo_cnn

echo "plotting"

module load python3/intel_3.6.3

python ./scripts/halo_cnn1d_r_plot.py

module unload python3/intel_3.6.3
