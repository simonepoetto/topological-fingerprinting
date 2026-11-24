"""
Export MSC FC matrices from pickle to individual .npz files
so that the Julia scaffold computation script can read them.

Output: data/MSC/fc/whole/npz/{sub}_s{sess}.npz  (key: 'fc')
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from topofinger import config
import pickle as pk
import numpy as np

data_dir = config.MSC
fc_path = os.path.join(data_dir, 'fc', 'whole', 'whole_corrs.pck')
out_dir = os.path.join(data_dir, 'fc', 'whole', 'npz')
os.makedirs(out_dir, exist_ok=True)

with open(fc_path, 'rb') as f:
    fc = pk.load(f)

count = 0
for sub in fc:
    for sess in fc[sub]:
        fname = f"{sub}_s{sess}.npz"
        out_path = os.path.join(out_dir, fname)
        np.savez_compressed(out_path, fc=fc[sub][sess])
        count += 1

print(f"Exported {count} FC matrices to {out_dir}")
