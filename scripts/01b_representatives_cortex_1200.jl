"""
H1 cycle representatives for HCP-1200 on the 300 CORTICAL regions only (session REST1_LR,
NoGSR_HighPassOnly, the same time series as the PhiR synergy of scripts/08_synergy_phiR.py; same
filtration as scripts/01_scaffolds_cortex_1200.jl). Used by the synergy analysis (Fig 4C, SI synergy
section): border links = cycle edges, internal links = missing chords.

Output: NoGSR_HighPassOnly/scaffolds/whole_cortex/representatives/<sub>_s0.pck
        (Python pickle: list of cycles, each a list of (i, j) edges, 1-based region indices)
Cohort: subjects with both LR sessions, as for the scaffolds. Resumable: existing outputs are skipped.

Usage:
  julia -t 8 scripts/01b_representatives_cortex_1200.jl
  julia -t 8 scripts/01b_representatives_cortex_1200.jl --test   # 1 subject into ./representatives_test, timed
"""

using Statistics
using Ripserer
using ProgressMeter
using NPZ
using Pickle

const N_CORTICAL = 300
const HCP1200 = get(ENV, "TOPOFINGER_HCP_ROOT", "./data/HCP/1200subjects/schaefer300")
const VARIANT = "NoGSR_HighPassOnly"

test = "--test" in ARGS
data_dir_ts = joinpath(HCP1200, VARIANT, "ts")
output_dir = test ? joinpath(pwd(), "representatives_test") : joinpath(HCP1200, VARIANT, "scaffolds", "whole_cortex", "representatives")
mkpath(output_dir)

files = readdir(data_dir_ts)
has(sub, n) = findfirst(f -> startswith(f, "$(sub)_rfMRI_REST$(n)_LR_Schaefer300+Tian_$(VARIANT)"), files)
subjects = sort([s for s in Set(split(f, '_')[1] for f in files) if has(s, 1) !== nothing && has(s, 2) !== nothing])
test && (subjects = subjects[1:1])
println("$(length(subjects)) subjects, $(Threads.nthreads()) threads, output $output_dir")

p = ProgressMeter.Progress(length(subjects))
t0 = time()
Threads.@threads for sub in subjects
    out = joinpath(output_dir, "$(sub)_s0.pck")
    if !isfile(out)
        ts = npzread(joinpath(data_dir_ts, files[has(sub, 1)]))["timeseries"]'   # (time, 316)
        dist = 1 .- cor(ts[:, 1:N_CORTICAL])                                       # cortical regions only
        dist = (dist .+ transpose(dist)) / 2
        d = ripserer(dist; reps=1, alg=:homology, verbose=false)
        store(out, [vertices.(representative(g)) for g in d[2]])
    end
    ProgressMeter.next!(p)
end
println("Done in $(round(time() - t0, digits=1)) s")
