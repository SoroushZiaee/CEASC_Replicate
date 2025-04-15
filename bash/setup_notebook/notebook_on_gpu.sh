#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter_gpu.out
#SBATCH --error=jupyter_gpu.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
#SBATCH --mem=20G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=eyakub@my.yorku.ca

# added 2 hrs back in bc was tough to get such job

echo "Start Installing and setup env"
source bash/setup_notebook/setup_env_node.sh

module list

pip freeze

# virtualenv --no-download $SLURM_TMPDIR/env
source /home/eyakub/projects/def-kohitij/eyakub/CEASC_Replicate/.venv/bin/activate

pip install --no-index --upgrade pip

echo "Installing requirements"
# pip install --no-index -r requirements.txt

echo "Env has been set up"

pip freeze

bash/setup_notebook/lab.sh
