"""
Sliding-window fingerprinting for HCP-1200 (Fig 2 success rate panel, window figures in the SI).
FC (1%) vs frequency scaffolds on the 300 cortical regions, for each window width and for
different sessions (REST1 vs REST2) or the same session (REST1 vs REST1).
For each bootstrap, one random window is drawn for each subject pair, as in fig2. In the same
session, a subject's two windows are always different, otherwise a window would be matched with itself.

Cohort: subjects with the modal number of windows at every width (995).
Output: results/panel_d_success_rate_bootstrap.pck (success rate, different sessions)
        results/windows_fingerprinting.pck (all four metrics)

Usage:
  python scripts/07_fingerprinting_windows_1200.py
  python scripts/07_fingerprinting_windows_1200.py --test   # 20 subjects, 1 width, checks the correlations, saves nothing
"""

import os, sys
import pickle as pk
from collections import Counter

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from topofinger import config
from topofinger.utils import threshold_matrix
from topofinger.fingerprinting import compute_effect_size, compute_success_rate

WIDTHS = [1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50]
N_BOOTSTRAP = 100
BASE = os.path.join(config.HCP_1200, config.DEFAULT_VARIANT)


def fc_path(width, sub):
    return os.path.join(BASE, 'fc', str(width), f'{sub}_fcs.npy')


def scaffold_path(width, sub):
    return os.path.join(BASE, 'scaffolds', 'windows_cortex', str(width), 'freq', f'{sub}_s01.npy')


def valid_subjects():
    """Subjects with FC and scaffolds and the modal number of windows, at every width."""
    valid = None
    for width in WIDTHS:
        subs = [f.split('_')[0] for f in sorted(os.listdir(os.path.join(BASE, 'fc', str(width))))]
        n_win = {s: np.load(fc_path(width, s), mmap_mode='r').shape[0] for s in subs if os.path.isfile(scaffold_path(width, s))}
        modal = Counter(n_win.values()).most_common(1)[0][0]
        keep = {s for s, n in n_win.items() if n == modal}
        valid = keep if valid is None else valid & keep
        print(f'Width {width}: {len(keep)} valid ({modal} windows), running intersection {len(valid)}')
    return sorted(valid)


def standardize(x):
    """Rows with zero mean and unit norm, so that a @ b.T gives their correlations."""
    x = x - x.mean(1, keepdims=True)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def load_width(width, subjects):
    """Standardized (n_subjects * n_windows, 300 * 300) arrays for each session."""
    fc, sc = ([], []), ([], [])
    for sub in tqdm(subjects, desc=f'Loading width {width}'):
        fcs = np.load(fc_path(width, sub))
        scs = np.load(scaffold_path(width, sub)).transpose(2, 0, 1).astype(np.float64)
        assert fcs.shape == scs.shape, (sub, fcs.shape, scs.shape)
        half = fcs.shape[0] // 2
        for k, windows in enumerate((slice(None, half), slice(half, None))):
            fc[k].append(standardize(np.array([threshold_matrix(f, 1).flatten() for f in fcs[windows]])))
            sc[k].append(standardize(scs[windows].reshape(half, -1)))
    return [np.concatenate(x) for x in fc], [np.concatenate(x) for x in sc], half


def bootstrap(corrs_fc, corrs_sc, n_subs, n_win, rng, rng_same=None):
    """Success rate and effect size, one random window per subject pair at each bootstrap.
    With rng_same (same session), the two windows of a subject are drawn different."""
    out = {k: [] for k in ('sr_fc', 'sr_sc', 'sr1_fc', 'sr2_fc', 'sr1_sc', 'sr2_sc', 'es_fc', 'es_sc')}
    base = np.arange(n_subs)[:, None] * n_win
    diag = np.arange(n_subs)
    for _ in range(N_BOOTSTRAP):
        win_rows = rng.integers(0, n_win, (n_subs, n_subs))
        win_cols = rng.integers(0, n_win, (n_subs, n_subs))
        if rng_same is not None:
            win_cols[diag, diag] = (win_rows[diag, diag] + rng_same.integers(1, n_win, n_subs)) % n_win
        rows, cols = base + win_rows, base.T + win_cols
        for name, corrs in (('fc', corrs_fc), ('sc', corrs_sc)):
            sim = corrs[rows, cols].astype(np.float64)
            sr1, sr2 = compute_success_rate(sim, kind='corr')
            out[f'sr1_{name}'].append(sr1)
            out[f'sr2_{name}'].append(sr2)
            out[f'sr_{name}'].append((sr1 + sr2) / 2)
            out[f'es_{name}'].append(compute_effect_size(sim, kind='corr'))
    return out


def summary(samples, sr1=None, sr2=None):
    res = {'mean': np.mean(samples), 'std': np.std(samples), 'bootstrap_samples': samples}
    if sr1 is not None:
        res.update(sr1_mean=np.mean(sr1), sr2_mean=np.mean(sr2))
    return res


def main():
    test = '--test' in sys.argv
    subjects = valid_subjects()
    widths = WIDTHS
    if test:
        subjects, widths = subjects[:20], [300]
    n_subs = len(subjects)
    rng = np.random.default_rng(0)
    rng_same = np.random.default_rng(1)  # separate stream, so the different-session draws do not depend on it
    results = {f'{m}_{c}_{s}': {} for m in ('success_rate', 'effect_size') for c in ('fc', 'sc') for s in ('different', 'same')}

    for width in widths:
        (fc0, fc1), (sc0, sc1), n_win = load_width(width, subjects)
        if test:
            ref = np.corrcoef(fc0.astype(np.float64), fc1.astype(np.float64))[fc0.shape[0]:, :fc0.shape[0]]
            print('Max difference with np.corrcoef:', np.abs(fc1 @ fc0.T - ref).max())
        for session, a_fc, a_sc, same in (('different', fc1, sc1, None), ('same', fc0, sc0, rng_same)):
            out = bootstrap(a_fc @ fc0.T, a_sc @ sc0.T, n_subs, n_win, rng, same)
            for c in ('fc', 'sc'):
                results[f'success_rate_{c}_{session}'][width] = summary(out[f'sr_{c}'], out[f'sr1_{c}'], out[f'sr2_{c}'])
                results[f'effect_size_{c}_{session}'][width] = summary(out[f'es_{c}'])
            print(f"Width {width}, {session} sessions: SR FC {np.mean(out['sr_fc']):.1f}, scaffolds {np.mean(out['sr_sc']):.1f}; "
                  f"d FC {np.mean(out['es_fc']):.2f}, scaffolds {np.mean(out['es_sc']):.2f}", flush=True)
        del fc0, fc1, sc0, sc1

    if test:
        return
    results.update(widths=WIDTHS, subjects=subjects, n_subs=n_subs, n_bootstrap=N_BOOTSTRAP)
    with open(os.path.join(config.RESULTS, 'windows_fingerprinting.pck'), 'wb') as f:
        pk.dump(results, f)
    with open(os.path.join(config.RESULTS, 'panel_d_success_rate_bootstrap.pck'), 'wb') as f:
        pk.dump({'success_rate_fc': results['success_rate_fc_different'], 'success_rate_sc': results['success_rate_sc_different'],
                 'widths': WIDTHS, 'n_subs': n_subs, 'n_bootstrap': N_BOOTSTRAP}, f)


if __name__ == '__main__':
    main()
