
# OpenPSS BenchMark Code and Datasets

This repository contains the code and experiments for the OpenPSS benchmark that accompany the "WOOIR: An Open Page Stream Segmentation Benchmark" paper.


## Installation
We recommend using Anaconda to install all the dependencies required for running the code in this package. The easiest way of doing this is by using the requirements.txt file provided in the repository. Assuming you hanve Anaconda installed, you can a fresh environment with the required packages using the following command.

``
conda create -n OPENPSS --file requirements.txt
``

## Resources
The datasets and model files and predictions can be downloaded via the below link.
https://surfdrive.surf.nl/files/index.php/s/afhAe6TuvC4eqir
You can download the folder and put it in the main file of the repository so that the notebooks and scripts can work with it appropriately.
The dataset also contains a short readme and a notebook exploring the contents of both datasets.

## Repository Structure

The repository has the following structure.
* model_train_code
   * Contains the scripts to train and run inference for the 4 neural methods. The other methods are not computationally expensive,
     and can be trained using the notebooks provided in the repository.
* Notebooks
   * Notebooks containing the experiments conducted in the paper, as well as a notebook containing a short data analysis for the LONG and SHORT datasets    
