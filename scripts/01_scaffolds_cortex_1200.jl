"""
Whole-scan frequency/persistence scaffolds for HCP-1200 on the 300 CORTICAL regions only
(Schaefer300; the 16 Tian subcortical regions in the time series are dropped before homology).

Output: <variant>/scaffolds/whole_cortex/{freq,pers}/<sub>_s<0|1>.npz
Resumable: existing outputs are skipped. Python picks these up automatically
(hcp_IO.load_scaffold_cortical) once the folder is complete; then rerun
`python scripts/06_similarity_matrices.py --force`.

Usage:
  julia -t 8 scripts/01_scaffolds_cortex_1200.jl                     # NoGSR_LowPassOnly
  julia -t 8 scripts/01_scaffolds_cortex_1200.jl NoGSR_NoFilt        # other variant
  julia -t 8 scripts/01_scaffolds_cortex_1200.jl NoGSR_LowPassOnly --test   # 1 subject into ./scaffold_test, timed
"""

using Statistics
using Ripserer
using ProgressMeter
using NPZ

const N_CORTICAL = 300
const HCP1200 = get(ENV, "TOPOFINGER_HCP_ROOT", "./data/HCP/1200subjects/schaefer300")

args = filter(a -> !startswith(a, "--"), ARGS)
variant = isempty(args) ? "NoGSR_LowPassOnly" : args[1]
test = "--test" in ARGS
gsr, bandpass = split(variant, "_"; limit=2)

data_dir = joinpath(HCP1200, variant)
data_dir_ts = joinpath(data_dir, "ts")
woutput_dir = test ? joinpath(pwd(), "scaffold_test") : joinpath(data_dir, "scaffolds", "whole_cortex")
mkpath(joinpath(woutput_dir, "freq"))
mkpath(joinpath(woutput_dir, "pers"))

files = readdir(data_dir_ts)
subjects = sort(collect(Set(split(f, '_')[1] for f in files)))
test && (subjects = subjects[1:1])
println("Variant $variant: $(length(subjects)) subjects, $(Threads.nthreads()) threads, output $woutput_dir")

function scaffold(ts)
    pms = cor(ts)
    n = size(pms, 2)
    dist = 1 .- pms
    dist = (dist .+ transpose(dist)) / 2
    d = ripserer(dist; reps=1, alg=:homology, verbose=false)
    p_sc = zeros(n, n)
    f_sc = zeros(n, n)
    for g in d[2]
        pers = persistence(g)
        for rep in representative(g)
            i, j = vertices(rep)
            p_sc[i, j] += pers; f_sc[i, j] += 1
            p_sc[j, i] += pers; f_sc[j, i] += 1
        end
    end
    return f_sc, p_sc
end

p = ProgressMeter.Progress(length(subjects))
t0 = time()
Threads.@threads for sub in subjects
    sess_files = String[]
    for n_sess in (1, 2)
        base = "$(sub)_rfMRI_REST$(n_sess)_LR_Schaefer300+Tian_$(gsr)_$(bandpass)"
        idx = findfirst(f -> startswith(f, base), files)
        idx === nothing && break
        push!(sess_files, files[idx])
    end
    if length(sess_files) == 2
        for (n_sess, fname) in enumerate(sess_files)
            out_f = joinpath(woutput_dir, "freq", "$(sub)_s$(n_sess - 1).npz")
            out_p = joinpath(woutput_dir, "pers", "$(sub)_s$(n_sess - 1).npz")
            (isfile(out_f) && isfile(out_p)) && continue
            ts = npzread(joinpath(data_dir_ts, fname))["timeseries"]'   # (time, 316)
            f_sc, p_sc = scaffold(ts[:, 1:N_CORTICAL])                    # cortical regions only
            npzwrite(out_f, f_sc)
            npzwrite(out_p, p_sc)
        end
    end
    ProgressMeter.next!(p)
end
println("Done in $(round(time() - t0, digits=1)) s")
