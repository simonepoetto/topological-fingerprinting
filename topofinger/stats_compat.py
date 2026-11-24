"""Compatibility helpers for pingouin, whose column names changed in 0.6
('p-unc' -> 'p_unc', 'p-corr' -> 'p_corr', 'p-GG-corr' -> 'p_GG_corr', ...)."""

_OLD_TO_NEW = {
    'p-unc': 'p_unc',
    'p-corr': 'p_corr',
    'p-GG-corr': 'p_GG_corr',
    'p-spher': 'p_spher',
    'W-spher': 'W_spher',
}


def normalize_pingouin(df):
    """Return df with old pingouin column names renamed to the new (>= 0.6) ones; no-op on new versions."""
    return df.rename(columns={k: v for k, v in _OLD_TO_NEW.items() if k in df.columns})
