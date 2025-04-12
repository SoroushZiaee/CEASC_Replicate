# CEASC_Replicate
NOTE: You will need to use download_uavdt.py and download_visdrone.py to access the datasets. Please alter the file paths in these scripts to ensure that datasets are downloaded an unzipped in an appropriate directory.

## Setup
### On Compute Canada
First load the following environment modules: 
```
module load StdEnv/2023 python/3.11.5 scipy-stack opencv/4.11.0
```

Then build your virtual environment and use requirements.txt to install dependencies.
```
virtualenv --no-download .venv
source .venv/bin/activate
pip install -r --no-index requirements.txt
```


### On a Different Cluster
Our code was built and run using python 3.11.5. For best compatibility, please use the closest possible version of python.

Build your virtual environment and use requirements2.txt to install dependencies.
```
virtualenv --no-download .venv
source .venv/bin/activate
pip install -r requirements2.txt
```
Note: Use with caution, we won't tell you how to manage your cluster 🙂

### Loading the Datasets 

## Training

## Evaluation

## Pre-Trained Models

