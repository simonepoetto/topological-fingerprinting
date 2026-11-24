"""Central data locations. Data live OUTSIDE the repo; override any entry with an environment variable."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DATA_ROOT = os.environ.get("TOPOFINGER_DATA_ROOT", os.path.join(_REPO_ROOT, "data"))

HCP_1200 = os.environ.get("TOPOFINGER_HCP_ROOT", os.path.join(DATA_ROOT, "HCP/1200subjects/schaefer300"))
HCP_100 = os.environ.get("TOPOFINGER_HCP100_ROOT", os.path.join(DATA_ROOT, "HCP/100unrelated"))
MSC = os.environ.get("TOPOFINGER_MSC_ROOT", os.path.join(DATA_ROOT, "MSC"))
DEFAULT_VARIANT = "NoGSR_LowPassOnly"
# HCP-1200 data have 316 regions (Schaefer300 + 16 Tian subcortical); the paper uses only the 300 cortical ones.
N_CORTICAL = 300
# HCP-100 Glasser data have 379 regions (360 cortical + 19 subcortical, rows 361-379); the paper uses the 360 cortical ones.
N_CORTICAL_GLASSER = 360

# HCP behavioral table, used in notebook 04. Keep outside the repo.
BEHAV_CSV = os.environ.get("TOPOFINGER_BEHAV_CSV", os.path.join(DATA_ROOT, "HCP/unrestricted_behavioral.csv"))

RESULTS = os.environ.get("TOPOFINGER_RESULTS", os.path.join(_REPO_ROOT, "results"))
# Raw/intermediate figure panels written by notebooks (not tracked).
FIGURES_RAW = os.environ.get("TOPOFINGER_FIGURES_RAW", os.path.join(_REPO_ROOT, "figures_raw"))

# Sub-folders of FIGURES_RAW that notebooks save into (created on import so savefig never fails on a fresh clone).
_FIGURES_RAW_SUBDIRS = ["", "final/fig1", "final/fig2", "final/fig3", "final/fig4", "fullhcp/fig4",
                        "figure1", "alternative_fc", "MSC", "supplementary", "synergy_stats_fix"]
for _d in _FIGURES_RAW_SUBDIRS:
    try:
        os.makedirs(os.path.join(FIGURES_RAW, _d), exist_ok=True)
    except OSError:
        pass
