"""
Table S4 (effects of preprocessing) on the 100 unrelated HCP subjects, Glasser parcellation,
360 cortical regions only (the 19 subcortical regions, rows 361-379, are excluded).

Time-series files: glasser379/ts/<sub>/rfMRI_REST<n>_<LR|RL>/Glasser360/TS_Glasser360S_<ts>.mat
  bp      bandpass, no GSR   (baseline; also the main Glasser results)
  gsr_bp  GSR + bandpass
  gsr     GSR, no temporal filtering
Rows (pairs):
  LR / RL                       bp
  no GSR / GSR                  bp  vs gsr_bp            (both bandpass)
  bandpass / no filtering       gsr_bp vs gsr            (both with GSR; separate low/high-pass not available)
Scaffolds: scripts/01_scaffolds_cortex_glasser.jl [--ts=<ts>] [--rl]. FC from the time series (NaN -> 0).

Output: results/preprocessing_glasser_360.csv and LaTeX rows printed to stdout.
Usage: python scripts/10_preprocessing_glasser.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from topofinger import config

import numpy as np
import pandas as pd
import scipy.io as sio
from collections import OrderedDict

from topofinger.hcp_IO import save_correlations_subjects_fast
from topofinger.utils import apply_threshold_to_nested_dict
from topofinger.fingerprinting import compute_success_rate, compute_effect_size

N = config.N_CORTICAL_GLASSER
G = os.path.join(config.HCP_100, "glasser379")
SUBJECTS = sorted(d for d in os.listdir(os.path.join(G, "ts")) if os.path.isdir(os.path.join(G, "ts", d)))

ROWS = OrderedDict([
    ("LR",                      ("bp", "LR")),
    ("RL",                      ("bp", "RL")),
    ("no GSR",                  ("bp", "LR")),
    ("GSR",                     ("gsr_bp", "LR")),
    ("bandpass (with GSR)",     ("gsr_bp", "LR")),
    ("no filtering (with GSR)", ("gsr", "LR")),
])


def load_fc(tsv, pe):
    fc = OrderedDict()
    for sub in SUBJECTS:
        fc[sub] = OrderedDict()
        for k in (1, 2):
            f = os.path.join(G, "ts", sub, f"rfMRI_REST{k}_{pe}", "Glasser360", f"TS_Glasser360S_{tsv}.mat")
            fc[sub][k - 1] = np.nan_to_num(np.corrcoef(sio.loadmat(f)["TS"][:N]))
    return fc


def load_scaffolds(tsv, pe):
    name = ("whole_cortex" if tsv == "bp" else f"whole_cortex_{tsv}") + ("_RL" if pe == "RL" else "")
    folder = os.path.join(G, "scaffolds", name, "freq")
    return OrderedDict((sub, OrderedDict((s, np.load(os.path.join(folder, f"{sub}_s{s}"))) for s in (0, 1))) for sub in SUBJECTS)


def metrics(sim):
    s1, s2 = compute_success_rate(sim)
    return (s1 + s2) / 2, compute_effect_size(sim)


rows, cache = [], {}
for label, key in ROWS.items():
    if key not in cache:
        fc = load_fc(*key)
        res = OrderedDict((f"FC {th}%", metrics(save_correlations_subjects_fast(apply_threshold_to_nested_dict(fc, th)))) for th in (100, 25, 1))
        res["Scaffolds"] = metrics(save_correlations_subjects_fast(load_scaffolds(*key)))
        cache[key] = res
    res = cache[key]
    row = {"row": label, "ts": key[0], "phase_encoding": key[1]}
    for k, (sr, d) in res.items():
        row[f"{k} SR"] = sr
        row[f"{k} d"] = round(d, 2)
    rows.append(row)
    print(f"{label:24s} " + "  ".join(f"{k}: SR {sr:5.1f} d {d:4.2f}" for k, (sr, d) in res.items()), flush=True)

out = os.path.join(config.RESULTS, "preprocessing_glasser_360.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved {out}\n\nLaTeX rows (SR):")
for r in rows:
    print(f"{r['row']} & {r['FC 100% SR']:.1f}\\% & {r['FC 25% SR']:.1f}\\% & {r['FC 1% SR']:.1f}\\% & \\textbf{{{r['Scaffolds SR']:.1f}\\%}} \\\\")
