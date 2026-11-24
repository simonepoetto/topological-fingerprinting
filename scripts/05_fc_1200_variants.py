"""
Compute FC (whole_corrs.pck) for 3 missing 1200-subject NoGSR variants:
  - NoGSR_HighPassOnly
  - NoGSR_NoFilt
  - NoGSR_Bandpass

Each variant has 4144 npz files in ts/ with naming:
  {sub}_rfMRI_REST{1,2}_{LR,RL}_Schaefer300+Tian_NoGSR_{variant}.npz

We concatenate REST1_LR + REST2_LR for session 0/1 and compute
per-subject FC matrices (Pearson correlation).
Output: fc/whole/whole_corrs.pck  (OrderedDict[sub][session] = 316x316 FC)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from topofinger import config
import sys
import numpy as np
import pickle as pk
from collections import OrderedDict
from tqdm import tqdm

BASE = config.HCP_1200
LR = "LR"
ATLAS = "Schaefer300"
GSR = "NoGSR"

VARIANTS = ["NoGSR_HighPassOnly", "NoGSR_NoFilt", "NoGSR_Bandpass"]

for variant in VARIANTS:
    bandpass = variant.split("_", 1)[1]  # e.g. "HighPassOnly"
    data_dir = os.path.join(BASE, variant)
    ts_dir = os.path.join(data_dir, "ts")
    fc_dir = os.path.join(data_dir, "fc", "whole")
    fc_path = os.path.join(fc_dir, "whole_corrs.pck")

    if os.path.exists(fc_path):
        print(f"[SKIP] {variant}: whole_corrs.pck already exists")
        continue

    os.makedirs(fc_dir, exist_ok=True)

    # Gather subject IDs
    files = sorted(os.listdir(ts_dir))
    subjects = sorted(set(f.split("_")[0] for f in files))
    print(f"\n[{variant}] Found {len(subjects)} unique subjects, {len(files)} files")

    whole_corrs = OrderedDict()
    skipped = 0

    for sub in tqdm(subjects, desc=variant):
        whole_corrs[sub] = OrderedDict()
        valid = True
        for sess_idx, rest_n in enumerate([1, 2]):
            pattern = f"{sub}_rfMRI_REST{rest_n}_{LR}_{ATLAS}+Tian_{GSR}_{bandpass}"
            matching = [f for f in files if f.startswith(pattern)]
            if not matching:
                valid = False
                break
            fpath = os.path.join(ts_dir, matching[0])
            data = np.load(fpath)
            ts = data["timeseries"]  # (316, 1200) or similar
            # FC = correlation across time (columns = time points for transposed)
            # ts shape is (regions, timepoints) based on earlier inspection
            fc = np.corrcoef(ts)  # (316, 316)
            whole_corrs[sub][sess_idx] = fc

        if not valid:
            del whole_corrs[sub]
            skipped += 1

    print(f"[{variant}] Computed FC for {len(whole_corrs)} subjects (skipped {skipped})")

    with open(fc_path, "wb") as f:
        pk.dump(whole_corrs, f)
    print(f"[{variant}] Saved {fc_path}")

print("\nDone!")
