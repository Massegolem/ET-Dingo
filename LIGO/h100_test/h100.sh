#!/bin/bash
#SBATCH --job-name "myjob"
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=01:00:00

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # one thread per IS worker
export CUDA_VISIBLE_DEVICES=0      

module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate $DATA/conda_envs/ET-DINGO                                 # one GPU is enough

dingo_pipe GW230824_033047.ini > log1.txt 2>&1
dingo_pipe GW230627_015337.ini > log2.txt 2>&1