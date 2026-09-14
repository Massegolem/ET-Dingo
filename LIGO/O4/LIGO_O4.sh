#!/bin/bash
#SBATCH --job-name=dingo_LIGO_O4
#SBATCH --account=p201368
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=22
#SBATCH --time=70:00:00
#SBATCH --output=LIGO_O4.out

module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate $DATA/conda_envs/ET-DINGO

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
time dingo_train --settings_file $DATA/ET-Dingo/ET-Dingo/LIGO/O4/train_settings.yaml --train_dir $DATA/models/LIGO_O4
