DATA_DIR=/scratch/tpausch/ASD/LIGO_O4_2
SETTINGS=$DATA_DIR/asd_dataset_settings.yaml

for i in 1 2 3 4 5; do
  ls $DATA_DIR/tmp/chunks/seg_*.pkl | xargs -P 24 -I{} \
  dingo_estimate_psds \
    --data_dir $DATA_DIR \
    --settings_file $SETTINGS \
    --time_segments_file {}

  echo "Attempt $i failed for $CHUNK, retrying in 15s..."
  sleep 1
done