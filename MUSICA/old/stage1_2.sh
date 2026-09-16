#!/bin/bash
#SBATCH --job-name=dingo_stage_1
#SBATCH --account=p201368
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=stage1_2.out

module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate $DATA/conda_envs/ET-DINGO

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
time dingo_train --checkpoint $DATA/models/stage1_2/model_latest.pt --train_dir $DATA/models/stage1_2
