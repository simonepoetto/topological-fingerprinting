"""
Pairwise PhiR synergy and MMI redundancy (Integrated Information Decomposition, hoi AtomsPhiID /
RedundancyphiID, Gaussian copula, tau = 1 TR) for the 1005-subject cohort, 300 cortical regions,
session REST1_LR.

Time series: NoGSR_HighPassOnly. Low-pass filtering (0.08 Hz at TR = 0.72 s) gives lag-1
autocorrelation ~0.99, which inflates the one-step time-delayed mutual information and removes the
cycle/synergy structure; high-pass-only series (lag-1 ~0.5) do not have this problem.
PhiR = sum of the synergistic and unique-transfer atoms (Mediano et al.).

Output: NoGSR_HighPassOnly/info/phiR_s0.npz with
  subjects (n,), syn (n, 44850) float32, red (n, 44850) float32  (condensed upper triangles, scipy squareform order)
Batches are checkpointed in NoGSR_HighPassOnly/info/checkpoints_phiR_s0/ (resumable).

Usage: python scripts/08_synergy_phiR.py [--batch-size 100] [--test]
"""

import os, sys, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from topofinger import config

import numpy as np
from hoi.metrics import AtomsPhiID, RedundancyphiID

PHIR_ATOMS = ['sts', 'u1ts', 'u2ts', 'stu2', 'stu1', 'rts', 'str', 'u1tu2', 'u2tu1']
VARIANT = "NoGSR_HighPassOnly"

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--test", action="store_true", help="first 4 subjects, output in ./phiR_test")
args = parser.parse_args()

ts_dir = os.path.join(config.HCP_1200, VARIANT, "ts")
out_dir = os.path.join(os.getcwd(), "phiR_test") if args.test else os.path.join(config.HCP_1200, VARIANT, "info")
ckpt_dir = os.path.join(out_dir, "checkpoints_phiR_s0")
os.makedirs(ckpt_dir, exist_ok=True)

# Cohort: subjects with both LR sessions in the main (low-pass) variant, as for the scaffolds
files = set(os.listdir(os.path.join(config.HCP_1200, "NoGSR_LowPassOnly", "ts")))
fname = lambda s, n, v: f"{s}_rfMRI_REST{n}_LR_Schaefer300+Tian_{v}.npz"
subjects = sorted(s for s in {f.split('_')[0] for f in files}
                  if fname(s, 1, "NoGSR_LowPassOnly") in files and fname(s, 2, "NoGSR_LowPassOnly") in files)
subjects = subjects[:4] if args.test else subjects
# hoi needs equal lengths within a batch: a few runs are shorter than 1200 TRs, so batch by length
length = {s: np.load(os.path.join(ts_dir, fname(s, 1, VARIANT)))["timeseries"].shape[1] for s in subjects}
batches = []
for n in sorted(set(length.values()), reverse=True):
    group = [s for s in subjects if length[s] == n]
    batches += [group[i:i + args.batch_size] for i in range(0, len(group), args.batch_size)]
print(f"{len(subjects)} subjects, {len(batches)} batches, output {out_dir}", flush=True)

t0 = time.time()
for b, subs in enumerate(batches):
    ckpt = os.path.join(ckpt_dir, f"batch_{b:03d}.npz")
    if os.path.exists(ckpt):
        continue
    ts = [np.load(os.path.join(ts_dir, fname(s, 1, VARIANT)))["timeseries"][:config.N_CORTICAL].astype(np.float64) for s in subs]
    data = np.stack(ts).transpose(2, 1, 0)   # (time, regions, subjects)
    syn = np.asarray(AtomsPhiID(data, verbose=False).fit(minsize=2, maxsize=2, atoms=PHIR_ATOMS), dtype=np.float32).T
    red = np.asarray(RedundancyphiID(data, verbose=False).fit(minsize=2, maxsize=2), dtype=np.float32).T
    np.savez(ckpt, subjects=np.array(subs), syn=syn, red=red)
    print(f"batch {b + 1}/{len(batches)} done, {time.time() - t0:.0f} s", flush=True)

parts = [np.load(os.path.join(ckpt_dir, f"batch_{b:03d}.npz")) for b in range(len(batches))]
out = os.path.join(out_dir, "phiR_s0.npz")
np.savez(out, subjects=np.concatenate([p["subjects"] for p in parts]),
         syn=np.concatenate([p["syn"] for p in parts]), red=np.concatenate([p["red"] for p in parts]))
print(f"Saved {out}")
