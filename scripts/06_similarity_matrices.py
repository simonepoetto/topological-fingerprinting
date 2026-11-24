"""
Compute scaffold and FC similarity matrices (subject x subject, test vs retest) for the
HCP-1200 filtering variants, using only the 300 cortical regions (subcortical Tian regions excluded).

Outputs (next to the inputs, originals untouched):
  <variant>/scaffolds/whole/scaffold_correlations_whole_cortex.npy
  <variant>/fc/whole/fc_correlations_whole_th{1,25,100}_cortex.npy

Scaffolds come from scaffolds/whole_cortex/freq if present (recomputed on 300 regions),
otherwise from the 316-region scaffolds cropped to 300 (see hcp_IO.load_scaffold_cortical).

Usage: python scripts/06_similarity_matrices.py [--force]
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from topofinger import config

import numpy as np

from topofinger.hcp_IO import load_scaffold_cortical, load_fc_cortical, save_correlations_subjects_fast
from topofinger.utils import apply_threshold_to_nested_dict

FORCE = '--force' in sys.argv
BASE = config.HCP_1200
N = config.N_CORTICAL
VARIANTS = ["NoGSR_LowPassOnly", "NoGSR_HighPassOnly", "NoGSR_NoFilt", "NoGSR_Bandpass"]

for variant in VARIANTS:
    vdir = os.path.join(BASE, variant)
    print(f"\n=== {variant} ===")

    sim_path = os.path.join(vdir, "scaffolds", "whole", "scaffold_correlations_whole_cortex.npy")
    if os.path.exists(sim_path) and not FORCE:
        print(f"[SKIP] scaffold similarity exists: {sim_path}")
    else:
        sc = load_scaffold_cortical(vdir, N)
        print(f"  {len(sc)} subjects, matrix {next(iter(sc.values()))[0].shape}")
        save_correlations_subjects_fast(sc, sim_path.replace('.npy', ''))
        print(f"  saved {sim_path}")
        del sc

    fc_paths = {th: os.path.join(vdir, "fc", "whole", f"fc_correlations_whole_th{th}_cortex.npy") for th in [1, 25, 100]}
    todo = [th for th, p in fc_paths.items() if FORCE or not os.path.exists(p)]
    if not todo:
        print("[SKIP] FC similarities exist")
        continue
    fc = load_fc_cortical(vdir, N)
    print(f"  FC: {len(fc)} subjects, matrix {next(iter(fc.values()))[0].shape}")
    for th in todo:
        save_correlations_subjects_fast(apply_threshold_to_nested_dict(fc, th), fc_paths[th].replace('.npy', ''))
        print(f"  saved {fc_paths[th]}")
    del fc

print("\nDone!")
