# split_segments.py
import pickle, numpy as np, yaml, os
from dingo.gw.noise.utils import get_time_segments

DATA_DIR = "/scratch/tpausch/ASD/LIGO_O3"          # must contain asd_dataset_settings.yaml
SETTINGS_FILE = os.path.join(DATA_DIR, "asd_dataset_settings.yaml")
N_WORKERS = 24                                # see note on picking this below

with open(SETTINGS_FILE) as f:
    settings = yaml.safe_load(f)["dataset_settings"]
run = settings["observing_run"]

time_segments = get_time_segments(settings)

# save the FULL segment dict too — dingo_merge_asd_datasets needs this later
seg_dir = os.path.join(DATA_DIR, "tmp", run)
os.makedirs(seg_dir, exist_ok=True)
with open(os.path.join(seg_dir, "psd_time_segments.pkl"), "wb") as f:
    pickle.dump(time_segments, f)

# split each detector's segments into N_WORKERS chunks
chunk_dir = os.path.join(DATA_DIR, "tmp", "chunks")
os.makedirs(chunk_dir, exist_ok=True)
for det, segs in time_segments.items():
    for i, sub in enumerate(np.array_split(segs, N_WORKERS)):
        if len(sub) == 0:
            continue
        with open(os.path.join(chunk_dir, f"seg_{det}_{i:03d}.pkl"), "wb") as f:
            pickle.dump({det: [tuple(s) for s in sub]}, f)