# OpenPSS BenchMark Code and Datasets.

This repository contains the code and experiments for the OpenPSS benchmark that accompany the "WOOIR: An Open Page Stream Segmentation Benchmark" paper.


## Installation
We recommend using Anaconda to install all the dependencies required for running the code in this package. The easiest way of doing this is by using the requirements.txt file provided in the repository. Assuming you hanve Anaconda installed, you can a fresh environment with the required packages using the following command.

``
conda create -n ENV_NAME --file requirements.txt
``

## Resources
``
experiment2.py --fusion_type FUSION_TYPE 
``

Where the fusion type is one of 'WIED-MM', 'EARLY', 'AUTOENCODER', 'LOGISTICREGRESSION'

