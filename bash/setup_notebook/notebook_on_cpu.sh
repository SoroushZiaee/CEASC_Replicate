#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter_cpu.out
#SBATCH --error=jupyter_cpu.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00
#SBATCH --mem=20G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

echo "Start Installing and setup env"
source bash/setup_notebook/setup_env_node.sh

module list

pip freeze

# virtualenv --no-download $SLURM_TMPDIR/env
source /home/soroush1/projects/def-kohitij/soroush1/CEASC_Replicate/.venv/bin/activate

pip install --no-index --upgrade pip

echo "Installing requirements"
# pip install --no-index -r requirements.txt

echo "Env has been set up"

pip freeze

bash/setup_notebook/lab.sh
