#!/usr/bin/env bash
#ASC --vanilla
#SBATCH --account=p201368
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=9:37:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=22      # up from 22
#SBATCH --threads-per-core=2
#SBATCH --mem=250G              # up from 190750M
#SBATCH --job-name=stage2_125_275
#SBATCH --output=stage2_125_275.out


module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate $DATA/conda_envs/ET-DINGO


CACHE=/local/$USER/$SLURM_JOB_ID
mkdir -p "$CACHE"
sed "s|/local/tpausch/JOBID_PLACEHOLDER|$CACHE|" $DATA/ET-Dingo/ET-Dingo/MUSICA/STAGE/train_settings_125_275.yaml > train_settings_job.yaml
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
time dingo_train --settings_file train_settings_job.yaml --train_dir $DATA/models/125_275

rm -rf "$CACHE" 