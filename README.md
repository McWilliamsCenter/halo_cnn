# halo_cnn
Repository for CNN models as presented in [A Robust and Efficient Deep Learning Method for Dynamical Mass Measurements of Galaxy Clusters](https://ui.adsabs.harvard.edu/abs/2019arXiv190205950H/abstract). This repository is meant to serve as an example of the code used to perform this deep learning analysis, not as a fully-developed utility for public use.

## Mocks
The mock cluster catalog generation script is [make_mocks.py](mocks/make_mocks.py). It uses halo data from the [MultiDark Planck 2 simulation](https://www.cosmosim.org/cms/simulations/mdpl2/) Rockstar catalog and a galaxy catalog generated using [UniverseMachine](https://ui.adsabs.harvard.edu/abs/2019MNRAS.tmp.1134B/abstract). The generated catalogs are stored as Catalog objects, detailed in [catalog.py](tools/catalog.py). For a full previously-generated mock catalog, reach out to the corresponding author at <mho1@andrew.cmu.edu>.

## Models
A brief tutorial on dataset preprocessing and model fitting is given in <tutorial.ipynb>. Data processing is handled by the HaloCNNDataManager class in [data.py](halo_cnn/data.py). Models are represented as the BaseHaloCNNRegressor class in [model.py](halo_cnn/model.py).

