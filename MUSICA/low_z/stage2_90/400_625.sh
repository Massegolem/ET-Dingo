#!/usr/bin/env bash
#ASC --vanilla
#SBATCH --account=p201368
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=14:24:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=22      # up from 22
#SBATCH --threads-per-core=2
#SBATCH --mem=250G              # up from 190750M
#SBATCH --job-name=stage2_400_625
#SBATCH --output=stage2_400_625.out


module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate $DATA/conda_envs/ET-DINGO


mkdir -p /local/tpausch/400_625

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
time dingo_train --settings_file train_settings_400_625.yaml --train_dir $DATA/models/400_625

rm -rf /local/tpausch/400_625




