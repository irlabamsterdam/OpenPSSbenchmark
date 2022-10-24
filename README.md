# ECIR-PSS-ReproducibilityResources

## Installation
We recommend using Anaconda to install all the dependencies required for running the code in this package. The easiest way of doing this is by using the requirements.txt file provided in the repository. Assuming you hanve Anaconda installed, you can a fresh environment with the required packages using the following command.

``
conda create -n ENV_NAME --file requirements.txt
``

This repository contais the code in the links the resources need to run the code used in this paper. The repository is
divided 3 main parts, corresponding to the three research questions in the paper.

## Overview
1. EXPERIMENT_1
2. EXPERIMENT_2
   - Because the models from EXPERIMENT_1 can take a significant time to evaluate when no GPU is available, we calculate the results here using the saved prediction dictionaries from experiment 1. However, we the predictions can also be obtained by following the guidelines for obtaining the predictions of the models, provided in the EXPERIMENT_1 folder.
4. EXPERIMENT_3

## Getting the resources
Because of the size of the dataset and the number of trained models, these are made available Zenodo and can be downloaded for usage by running 
the 'download_resources.sh' script.

## Experiment 1 Instructions
The scripts present in the EXPERIMENT_1 folder are all structured in a similar manner, and can be run using the same command line arguments. Below are some small examples on how to run these models from EXPERIMENT_1


Although all of the models will produce the ``predictions.json`` file required to evaluate the model, this can take a significant amount of time for some of the models, especially the GUHA-TXT and WIED-IMG models. For this reason, we have also provided the predictions of the output models in the ``model_outputs`` folder. You can also follow the scripts from EXPERIMENT_1 and run the predictions manually.
