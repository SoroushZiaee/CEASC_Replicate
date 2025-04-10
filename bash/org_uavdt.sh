#!/bin/bash
#SBATCH --job-name=org_uavdt
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=eyakub@my.yorku.ca
#SBATCH --mail-type=ALL

# navigate to the location this is being run and make log directory if it doesn't already exist
cd ~/projects/def-kohitij/eyakub/CEASC_Replicate # make sure to change this for the current user
mkdir -p logs 

# load the environment
module --force purge
module load StdEnv/2023
module load python/3.11.5 scipy-stack opencv/4.11.0
source .venv/bin/activate # make sure this connects to a good environment in your directory

# run the script to organize the train and test subsets on different nodes -- make sure to set these directories appropriately
srun --exclusive -N1 -n1 python scripts/UAVDT2COCO.py /home/eyakub/scratch/CEASC_replicate/UAV-benchmark-M /home/eyakub/scratch/CEASC_replicate/UAV-benchmark-M/train_anno.json testset
srun --exclusive -N1 -n1 python scripts/UAVDT2COCO.py /home/eyakub/scratch/CEASC_replicate/UAV-benchmark-M /home/eyakub/scratch/CEASC_replicate/UAV-benchmark-M/test_anno.json trainset

wait

