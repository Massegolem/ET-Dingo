DATA_DIR=/scratch/tpausch/ASD/LIGO_O3
SETTINGS=$DATA_DIR/asd_dataset_settings.yaml

while true; do
  timeout 3h bash -c '
  ls "$1"/tmp/chunks/seg_*.pkl | xargs -P 24 -I{} \
    dingo_estimate_psds \
      --data_dir "$1" \
      --settings_file "$2" \
      --time_segments_file {}
' _ "$DATA_DIR" "$SETTINGS"
  status=$?

  if [ $status -eq 0 ]; then
    echo "All segments completed successfully."
    break
  fi

  echo "3h window elapsed (exit $status) — restarting, idempotent skip-if-exists will resume remaining segments."
done