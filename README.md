# halo_cnn
Repository for CNN models for galaxy cluster mass estimation as presented in:
* [A Robust and Efficient Deep Learning Method for Dynamical Mass Measurements of Galaxy Clusters](https://ui.adsabs.harvard.edu/abs/2019arXiv190205950H/abstract).
* [Approximate Bayesian Uncertainties on Deep Learning Dynamical Mass Estimates of Galaxy Clusters](https://ui.adsabs.harvard.edu/abs/2021ApJ...908..204H/abstract)
* [The dynamical mass of the Coma cluster from deep learning](https://ui.adsabs.harvard.edu/abs/2022NatAs...6..936H/abstract)
* [Benchmarks and Explanations for Deep Learning Estimates of X-ray Galaxy Cluster Masses](https://ui.adsabs.harvard.edu/abs/2023arXiv230300005H/abstract)
This repository is meant to serve as an example of the code used to perform this deep learning analysis, not as a fully-developed utility for public use.

## Mocks
The mock cluster catalog generation script is [make_mocks.py](mocks/make_mocks.py). It uses halo data from the [MultiDark Planck 2 simulation](https://www.cosmosim.org/cms/simulations/mdpl2/) Rockstar catalog and a galaxy catalog generated using [UniverseMachine](https://ui.adsabs.harvard.edu/abs/2019MNRAS.tmp.1134B/abstract). The generated catalogs are stored as Catalog objects, detailed in [catalog.py](tools/catalog.py). For a full previously-generated mock catalog, reach out to the corresponding author at <matthew.annam.ho@gmail.com>.

## Models
Data processing is handled by the HaloCNNDataManager classes in [data](halo_cnn/data). Models are represented as the BaseHaloCNNRegressor class in [model](halo_cnn/model).

