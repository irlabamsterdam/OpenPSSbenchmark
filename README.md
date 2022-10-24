# ECIR-PSS-ReproducibilityResources

## Installation
We recommend using Anaconda to install all the dependencies required for running the code in this package. The easiest way of doing this is py using the requirements.txt / requirements.yaml files provided in the repository. Assuming you hanve Anaconda installed, you can a fresh environment with the required packages using the following command.

``
conda create -n ENV_NAME --file requirements.txt
``

This repository contais the code in the links the resources need to run the code used in this paper. The repository is
divided 3 main parts, corresponding to the three research questions in the paper.

1. EXPERIMENT_1
2. EXPERIMENT_2
   - Because the models from EXPERIMENT_1 can take a significant time to evaluate when no GPU is available, we calculate the results here using the saved prediction dictionaries from experiment 1. However, we the predictions can also be obtained by following the guidelines for obtaining the predictions of the models, provided in the EXPERIMENT_1 folder.
4. EXPERIMENT_3
  
Because of the size of the dataset and the number of trained models, these are made available through separate links.

## Dataset
The datasets, models and output vectors are made available through Zenodo. We have provided the images as 224 by 224, as this is the image size used by the models in this paper, and it significantly reduces the size of the datasets, making at more suitable for reproducibility.
## Trained Models
 - 
 - The models and data can be downloaded manually, but can also be downloaded using the scripts in 'dataset' and 'trained_models'

## Experiment 1 Instructions
The scripts present in the EXPERIMENT_1 folder are all structured in a similar manner, and can be run using the same command line arguments. Below are some small examples on how to run these models from EXPERIMENT_1
