#!/usr/bin/env bash
#ASC --vanilla
#SBATCH --account=p201368
#SBATCH -p zen4_0768
#SBATCH --qos=zen4_0768
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=180     # up from 22
#SBATCH --threads-per-core=2
#SBATCH --mem=250G              # up from 190750M
#SBATCH --job-name=stage2_waves
#SBATCH --output=stage2_waves.out


module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate $DATA/conda_envs/ET-DINGO


dingo_generate_dataset --settings waveform_dataset_settings_50_100.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_50_100.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_50_125.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_50_125.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_75_150.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_75_150.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_100_175.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_100_175.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_125_200.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_125_200.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_125_275.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_125_275.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_175_325.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_175_325.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_225_375.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_225_375.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_275_425.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_275_425.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_325_475.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_325_475.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_375_525.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_375_525.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_400_625.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_400_625.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_475_700.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_475_700.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_525_825.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_525_825.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_600_975.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_600_975.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_750_1000.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_750_1000.hdf5
dingo_generate_dataset --settings waveform_dataset_settings_900_1000.yaml --out_file /scratch/fs201368/tp41139/waveforms/waveform_dataset_900_1000.hdf5