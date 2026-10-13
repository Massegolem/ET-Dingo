#!/usr/bin/env bash
#ASC --vanilla
#SBATCH --account=p201368
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=22      # up from 22
#SBATCH --threads-per-core=2
#SBATCH --mem=250G              # up from 190750M




module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate $DATA/conda_envs/ET-DINGO

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
time dingo_train --settings_file $DATA/ET-Dingo/ET-Dingo/LIGO/O4/T4/train_settings2.yaml --train_dir $DATA/models/LIGO_O4_T4

