#!/usr/bin/env bash
#ASC --vanilla
#SBATCH --account=p201368
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=22      # up from 22
#SBATCH --threads-per-core=2
#SBATCH --mem=400G              # up from 190750M
#SBATCH --job-name=dingo_LIGO_O4_T8_continue
#SBATCH --output=LIGO_O4_T8_continue.out



export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate $DATA/conda_envs/ET-DINGO


mkdir -p /local/tpausch/LIGO_O4_T8_low_mass

#python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#time dingo_train --settings_file $DATA/ET-Dingo/ET-Dingo/LIGO/O3/T8/train_settings.yaml --train_dir $DATA/models/LIGO_O3_T8

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
time dingo_train --checkpoint $DATA/models/LIGO_O3_T8/model_latest.pt --train_dir $DATA/models/LIGO_O3_T8


rm -rf /local/tpausch/LIGO_O4_T8_low_mass
