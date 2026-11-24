"""
Whole-scan frequency/persistence scaffolds for the HCP-100 Glasser parcellation on the
360 CORTICAL regions only (the 19 subcortical regions, rows 361-379 of TS_Glasser360S_bp.mat, are dropped).

Output: glasser379/scaffolds/whole_cortex/{freq,pers}/<sub>_s<0|1>  (no extension, as the original Glasser scaffolds)
Resumable. Then run: python scripts/06b_similarity_matrices_glasser.py --force

Usage:
  julia -t 8 scripts/01_scaffolds_cortex_glasser.jl          # all 100 subjects
  julia -t 8 scripts/01_scaffolds_cortex_glasser.jl --test   # 1 subject into ./scaffold_test_glasser, timed
  julia -t 8 scripts/01_scaffolds_cortex_glasser.jl --ts=gsr_bp   # other preprocessing: TS_Glasser360S_<ts>.mat
                                                                   # (bp = bandpass, no GSR [default]; gsr_bp; gsr = GSR, no filtering)
                                                                   # output whole_cortex_<ts> (whole_cortex for bp)
  julia -t 8 scripts/01_scaffolds_cortex_glasser.jl --rl          # RL phase-encoding runs (output suffix _RL)
"""

using Statistics
using Ripserer
using ProgressMeter
using NPZ
using MAT

const N_CORTICAL = 360
const HCP100 = get(ENV, "TOPOFINGER_HCP100_ROOT", "./data/HCP/100unrelated")

test = "--test" in ARGS
ts_arg = filter(a -> startswith(a, "--ts="), ARGS)
tsv = isempty(ts_arg) ? "bp" : split(ts_arg[1], "=", limit=2)[2]
pe = "--rl" in ARGS ? "RL" : "LR"
outname = (tsv == "bp" ? "whole_cortex" : "whole_cortex_$(tsv)") * (pe == "RL" ? "_RL" : "")
data_dir = joinpath(HCP100, "glasser379", "ts")
woutput_dir = test ? joinpath(pwd(), "scaffold_test_glasser") : joinpath(HCP100, "glasser379", "scaffolds", outname)
mkpath(joinpath(woutput_dir, "freq"))
mkpath(joinpath(woutput_dir, "pers"))

subjects = sort(filter(d -> isdir(joinpath(data_dir, d)), readdir(data_dir)))
test && (subjects = subjects[1:1])
println("Glasser cortical ($tsv, $pe): $(length(subjects)) subjects, $(Threads.nthreads()) threads, output $woutput_dir")

function scaffold(ts)
    pms = cor(ts)
    n = size(pms, 2)
    dist = 1 .- pms
    dist = (dist .+ transpose(dist)) / 2
    dist = replace(dist, NaN => 1.5)   # as in the original Glasser pipeline
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
    for sess in (1, 2)
        out_f = joinpath(woutput_dir, "freq", "$(sub)_s$(sess - 1)")
        out_p = joinpath(woutput_dir, "pers", "$(sub)_s$(sess - 1)")
        (isfile(out_f) && isfile(out_p)) && continue
        path = joinpath(data_dir, sub, "rfMRI_REST$(sess)_$(pe)", "Glasser360", "TS_Glasser360S_$(tsv).mat")
        isfile(path) || continue
        ts = transpose(matread(path)["TS"])          # (time, 379)
        f_sc, p_sc = scaffold(ts[:, 1:N_CORTICAL])   # cortical regions only
        npzwrite(out_f, f_sc)
        npzwrite(out_p, p_sc)
    end
    ProgressMeter.next!(p)
end
println("Done in $(round(time() - t0, digits=1)) s")
