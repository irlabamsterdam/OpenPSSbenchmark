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
the 'download_resources.sh' script. The README file in the resources folder contains more information on the precise contents of the resource folder.

## Experiment 1 Instructions
The scripts present in the EXPERIMENT_1 folder are all structured in a similar manner, and can be run using the same command line arguments. Below are some small examples on how to run these models from EXPERIMENT_1. By default, all of the scripts in this research use the pretrained models on the C1 and C2 datasets, however, by using the ``from_scratch`` parameter, the models can be re-trained. However be aware that this can take a very long time, it took as about 2 days on a GPU to train the WIED-IMG model on the C1 dataset.

To automatically get all the scores, run the 'experimen1.sh' script, otherwise run the different models seperately, using the following structure:
`` MODEL_FILE.py --train_dataset C1 --test_dataset C1``. This will print the scores for the model given by the MODEL_FILE name on the C1 dataset and save the output vectors. Running experiment1.sh will run the models with preloaded predictions.


Although all of the models will produce the ``predictions.json`` file required to evaluate the model, this can take a significant amount of time for some of the models, especially the GUHA-TXT and WIED-IMG models. For this reason, we have also provided the predictions of the output models in the ``model_outputs`` folder. You can also follow the scripts from EXPERIMENT_1 and run the predictions manually.

## Experiment 2 Instructions
Experiment 2 is an extension of experiment 1 where we use the results we got from experiment 1 and experiment with test data from a different source. To run this experiment 'from scratch' you can run all the models in experiment 1 for both C1 and C2 with prediction saving enabled, and then run the script `experiment2.py` in the experiment 2 folder to get all the results.

## Experiment 3 Instructions

Experiment 3 only uses the vectors extracted from the experiment 1 models and when using these, the fusion architectures are quite small and can easily be trained from scratch, and thus we also do not provide the predictions for these models in files. Please refer to the `experiment3.sh` script to see how to run the fusion architectures.
