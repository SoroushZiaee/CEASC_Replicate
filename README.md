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
pip install --no-index -r requirements.txt
```
Follow the [instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html) for using MMDetection as a third-party package to install MMDetection.

### On a Different Cluster
Our code was built and run using python 3.11.5. For best compatibility, please use the closest possible version of python.

Build your virtual environment and use requirements2.txt to install dependencies.
```
virtualenv --no-download .venv
source .venv/bin/activate
pip install -r requirements2.txt
```
Follow the [instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html) for using MMDetection as a third-party package to install MMDetection.

Note: Use with caution, we won't tell you how to manage your cluster 🙂

### With Conda

#### Create the Conda Environment
Use the provided `environment.yaml` to create the environment:
```bash
conda env create -f environment.yaml
```

#### Activate the Environment
```bash
conda activate ceasc-env
```

#### 🧪 Verify Installation
You can verify that core packages are installed properly by running:
```bash
python -c "import torch, mmdet, mmcv, mmengine; print('✅ Environment is ready!')"
```
### Loading the Datasets 

#### VisDrone
Run the following code to download the VisDrone dataset. Make sure to edit the script to reflect your directory organization and desired output location for 4 directories: VisDrone2019-DET-test-challenge, VisDrone2019-DET-test-dev, VisDrone2019-DET-train, VisDrone2019-DET-val.

```
python scripts/download_visdrone.py
```
#### UAVDT
Run the following code to download the UAVDT dataset and make a meta file to specify training, validation, and testing images. Make sure to edit the script to reflect your directory organization and desired output location for 2 directories: UAV-benchmark-M and UAV-benchmark-MOTD_v1.0. 

```
python scripts/download_uavdt.py
python scripts/make_meta_uavdt.py --root_dir <insert path to UAV-benchmark-M directory>
```


## Training

## Evaluation

## Pre-Trained Models

