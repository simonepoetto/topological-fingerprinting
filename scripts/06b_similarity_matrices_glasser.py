"""
Scaffold and FC similarity matrices for the HCP-100 Glasser parcellation using only the
360 cortical regions (the 19 subcortical regions, rows 361-379, are excluded).

Outputs (next to the inputs, originals untouched):
  glasser379/scaffolds/whole/scaffold_correlations_whole_cortex.npy
  glasser379/fc/whole/fc_correlations_whole_th{1,25,100}_cortex.npy

Scaffolds come from glasser379/scaffolds/whole_cortex/freq if complete (recomputed on 360 regions,
scripts/01_scaffolds_cortex_glasser.jl), otherwise from the 379-region scaffolds cropped to 360.

Usage: python scripts/06b_similarity_matrices_glasser.py [--force]
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from topofinger import config

from topofinger.hcp_IO import load_scaffold, load_scaffold_cortical, load_fc_cortical, save_correlations_subjects_fast
from topofinger.utils import apply_threshold_to_nested_dict

FORCE = '--force' in sys.argv
N = config.N_CORTICAL_GLASSER
gdir = os.path.join(config.HCP_100, "glasser379")

sim_path = os.path.join(gdir, "scaffolds", "whole", "scaffold_correlations_whole_cortex.npy")
if os.path.exists(sim_path) and not FORCE:
    print(f"[SKIP] scaffold similarity exists: {sim_path}")
else:
    sc = load_scaffold_cortical(gdir, N, loader=load_scaffold)
    print(f"  {len(sc)} subjects, matrix {next(iter(sc.values()))[0].shape}")
    save_correlations_subjects_fast(sc, sim_path.replace('.npy', ''))
    print(f"  saved {sim_path}")

fc = load_fc_cortical(gdir, N)
print(f"  FC: {len(fc)} subjects, matrix {next(iter(fc.values()))[0].shape}")
for th in [1, 25, 100]:
    p = os.path.join(gdir, "fc", "whole", f"fc_correlations_whole_th{th}_cortex.npy")
    if os.path.exists(p) and not FORCE:
        print(f"[SKIP] {p}")
        continue
    save_correlations_subjects_fast(apply_threshold_to_nested_dict(fc, th), p.replace('.npy', ''))
    print(f"  saved {p}")
print("Done!")
