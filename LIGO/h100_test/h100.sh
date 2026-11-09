#!/bin/bash
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # one thread per IS worker
export CUDA_VISIBLE_DEVICES=0                                       # one GPU is enough

dingo_pipe GW230824_033047.ini > log1.txt 2>&1
dingo_pipe GW230627_015337.ini > log2.txt 2>&1