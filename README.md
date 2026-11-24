# Topological Fingerprinting

Code accompanying ***The Topological Architecture of Brain Identity***
(Simone Poetto, Haily Merritt, Andrea Santoro, Giovanni Rabuffo, Karolina Finc, Demian Battaglia, Francesco Vaccarino, Manish Saggar, Andrea Brovelli, Giovanni Petri).

This repository provides the Python and Julia code for computing **homological scaffolds** (persistent $H_1$ cycles from resting-state fMRI functional connectivity), performing **individual fingerprinting** across cohorts, atlases, scan lengths, and preprocessing pipelines, and analyzing **synergistic and redundant information dynamics** ($\Phi_R$) on cycle borders and interiors.

## Repository Structure

```text
topofinger/   Core Python package (config, I/O, fingerprinting, synergy/PhiID, plotting)
scripts/      Numbered batch pipeline (Julia scaffold & cycle extraction 01–03; Python 04–10)
notebooks/    Analysis and figure-generation notebooks (01a–05, fig1–fig4, S_*)
env.yml       Conda environment (Python 3.11) and required Julia packages
```

## Installation

### 1. Python environment
```bash
conda env create -f env.yml
conda activate topofinger
```

### 2. Julia packages (for homological scaffolds)
Persistent homology ($H_1$ scaffolds and representative cycles) is computed in Julia ($\ge 1.9$) with [Ripserer.jl](https://github.com/mtsch/Ripserer.jl):
```julia
using Pkg
Pkg.add(["Ripserer", "NPZ", "MAT", "Pickle", "ProgressMeter", "DelimitedFiles", "Statistics"])
```

## Data Configuration

Raw neuroimaging datasets are not included in this repository:
- **Human Connectome Project (HCP)** Young Adult (1200-subject release; 1005 subjects with complete test-retest, 300 cortical Schaefer regions; plus 100 unrelated benchmark across AAL90, Schaefer100, Shen268, Schaefer300, Glasser360): available from [humanconnectome.org](https://www.humanconnectome.org/study/hcp-young-adult) under WU-Minn HCP data use terms.
- **Midnight Scan Club (MSC)** (10 subjects $\times$ 10 sessions, Schaefer400-Yeo17): available on OpenNeuro ([ds000224](https://openneuro.org/datasets/ds000224)).

All paths are configured centrally in `topofinger/config.py` and can be overridden with environment variables:
```bash
export TOPOFINGER_DATA_ROOT=/path/to/data
# Optional granular overrides:
# export TOPOFINGER_HCP_ROOT=/path/to/data/HCP/1200subjects/schaefer300
# export TOPOFINGER_HCP100_ROOT=/path/to/data/HCP/100unrelated
# export TOPOFINGER_MSC_ROOT=/path/to/data/MSC
# export TOPOFINGER_BEHAV_CSV=/path/to/unrestricted_behavioral.csv
# export TOPOFINGER_RESULTS=/path/to/results
# export TOPOFINGER_FIGURES_RAW=/path/to/figures_raw
```

## Pipeline Overview

### Batch scripts (`scripts/`)
| Script | Language | Purpose |
|---|---|---|
| `01_scaffolds_cortex_1200.jl` | Julia | Homological scaffolds on HCP-1200 (300 cortical Schaefer ROIs) |
| `01_scaffolds_cortex_glasser.jl` | Julia | Homological scaffolds on HCP-100 Glasser (360 cortical ROIs) |
| `01b_representatives_cortex_1200.jl` | Julia | Persistent $H_1$ representative cycles on HCP-1200 (high-pass data) |
| `02_scaffolds_windows_1200.jl` | Julia | Windowed scaffolds across scan lengths (HCP-1200) |
| `03_scaffolds_filtering_1200.jl` | Julia | Scaffolds across temporal filtering variants (HCP-1200) |
| `04_export_msc_fc_to_npz.py` | Python | Export MSC functional connectivity matrices |
| `05_fc_1200_variants.py` | Python | Compute FC matrices across filtering variants (HCP-1200) |
| `06_similarity_matrices.py` | Python | Cross-session similarity matrices for HCP-1200 (FC and scaffolds) |
| `06b_similarity_matrices_glasser.py` | Python | Cross-session similarity matrices for Glasser 360 |
| `07_fingerprinting_windows_1200.py` | Python | Scan-length and intra/inter-session fingerprinting metrics |
| `08_synergy_phiR.py` | Python | Integrated information decomposition ($\Phi_R$ synergy/redundancy) |
| `10_preprocessing_glasser.py` | Python | Preprocessing comparison on Glasser 360 |

### Notebooks (`notebooks/`)
- **Core pipeline & intermediates:** `01a_scaffolds_hcp_julia.ipynb`, `01b_scaffolds_msc_julia.ipynb`, `01d_scaffolds_representatives_julia.ipynb`, `02_fc_from_timeseries.ipynb`, `03_distance_matrices.ipynb`, `04_edge_importance.ipynb`, `05_internal_links.ipynb`
- **Main figures:** `fig1.ipynb`, `fig2.ipynb`, `fig3.ipynb`, `fig4.ipynb`
- **Supplementary analyses:** `S_alternative_fc.ipynb`, `S_filtering.ipynb`, `S_msc.ipynb`, `S_parameter_sensitivity.ipynb`, `S_synergy_stats.ipynb`

> **Note on previous version:** The initial minimal preprint release is preserved on the `v1-preprint` git tag.

## License

Released under the [MIT License](LICENSE).
