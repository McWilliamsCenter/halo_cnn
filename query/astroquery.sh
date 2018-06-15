# interact --egress --ntasks-per-node=5

cd /home/mho1/halo_cnn/query

module load anaconda2/5.1.0

source activate astroq

python query_cosmosim.py

source deactivate astroq
