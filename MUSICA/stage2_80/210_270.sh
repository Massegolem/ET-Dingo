#!/usr/bin/env bash
#ASC --vanilla
#SBATCH --account=p201368
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=15:22:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=22      # up from 22
#SBATCH --threads-per-core=2
#SBATCH --mem=250G              # up from 190750M
#SBATCH --job-name=stage2_210_270
#SBATCH --output=stage2_210_270.out


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate $DATA/conda_envs/ET-DINGO


mkdir -p /local/tpausch/210_270

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
time dingo_train --settings_file train_settings_210_270.yaml --train_dir $DATA/models/stage2_80/210_270

rm -rf /local/tpausch/210_270




