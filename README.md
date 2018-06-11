# halo_cnn
Workspace for applying Convolutional Neural Networks to galaxy cluster mass measurements

## Project is under development
I'm planning to use this git repository just for my own use. Mostly to help with version control. Also to take advantage of McWilliams Center private repositories :)

## ML Pipeline
Given a model_name,
1. model_name_data.py : Preprocess data and evaluate data characteristics
2. model_name_ml.py : Run ML and plot training statistics
3. model_name_plot.py : Plot predictions and results

## Main Directories
* bash - bash scripts intended for 'ease-of-use.' Manages loading correct directories
* jobs - jobs to be submitted to BRIDGES queue
* notebooks - jupyter workspaces
* old - old stuff
* scripts - Python code for setting up, running ML model
* tools - General use tools (figure plotting, data loading, etc.)
