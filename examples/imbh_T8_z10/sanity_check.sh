#!/bin/bash
#SBATCH --job-name=dingo_sanity_check
#SBATCH --account=p201368
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=sanity_%j.out

module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate $DATA/conda_envs/ET-DINGO

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
timeout 300 dingo_train --settings_file $DATA/ET-Dingo/ET-Dingo/examples/imbh_T8_z10/train_settings_mass2_8.yaml --train_dir $SCRATCH/models/sanity_test
