"""
Compute scaffolds for HCP 1200-subject filtering variants using Julia.
Runs Julia as a subprocess with 36 threads.
"""

using DelimitedFiles
using Statistics
using Ripserer
using ProgressMeter
using NPZ

function compute_scaffolds_for_variant(variant_name, bandpass_tag)
    data_dir = joinpath(get(ENV, "TOPOFINGER_HCP_ROOT", "./data/HCP/1200subjects/schaefer300"), variant_name) * "/"
    data_dir_ts = joinpath(data_dir, "ts")
    woutput_dir = joinpath(data_dir, "scaffolds", "whole")
    mkpath(joinpath(woutput_dir, "freq"))
    mkpath(joinpath(woutput_dir, "pers"))

    lr = "LR"
    atlas = "Schaefer300"
    gsr = "NoGSR"

    files = readdir(data_dir_ts)
    subjects = sort(collect(Set(split(f, '_')[1] for f in files)))

    # Count already done
    existing_freq = length(readdir(joinpath(woutput_dir, "freq")))
    println("[$variant_name] $(length(subjects)) subjects, $existing_freq scaffolds already computed")

    if existing_freq >= 2 * length(subjects)
        println("[$variant_name] All scaffolds already computed, skipping.")
        return
    end

    n = length(subjects)
    p = ProgressMeter.Progress(n, desc="[$variant_name] ")

    Threads.@threads for sub in subjects
        sess_files = String[]
        valid_subject = true
        for n_sess in (1, 2)
            pattern_base = "$(sub)_rfMRI_REST$(n_sess)_$(lr)_$(atlas)+Tian_$(gsr)_$(bandpass_tag)"
            found_idx = findfirst(f -> startswith(f, pattern_base), files)
            if found_idx === nothing
                valid_subject = false
                break
            end
            push!(sess_files, files[found_idx])
        end

        if !valid_subject
            ProgressMeter.next!(p)
            continue
        end

        for (n_sess, fname) in enumerate(sess_files)
            session_idx = n_sess - 1
            file_path = joinpath(data_dir_ts, fname)

            out_freq_path = joinpath(woutput_dir, "freq", "$(sub)_s$(session_idx).npz")
            out_pers_path = joinpath(woutput_dir, "pers", "$(sub)_s$(session_idx).npz")

            if isfile(out_freq_path) && isfile(out_pers_path)
                continue
            end

            data = npzread(file_path)
            ts = data["timeseries"]'

            pms = cor(ts)
            num_regions = size(pms, 2)

            dist_corrs = 1 .- pms
            dist_corrs = (dist_corrs .+ transpose(dist_corrs)) / 2

            d = ripserer(dist_corrs; reps=1, alg=:homology, verbose=false)

            p_scaffold_b = zeros(num_regions, num_regions)
            f_scaffold_b = zeros(num_regions, num_regions)

            for l in 1:length(d[2])
                g = d[2][l]
                pers = persistence(g)
                for rep in representative(g)
                    i, j = vertices(rep)
                    p_scaffold_b[i, j] += pers
                    f_scaffold_b[i, j] += 1
                    p_scaffold_b[j, i] += pers
                    f_scaffold_b[j, i] += 1
                end
            end

            npzwrite(out_freq_path, f_scaffold_b)
            npzwrite(out_pers_path, p_scaffold_b)
        end

        ProgressMeter.next!(p)
    end
    println("[$variant_name] Done!")
end

# Run all 3 variants
variants = [
    ("NoGSR_HighPassOnly", "HighPassOnly"),
    ("NoGSR_NoFilt", "NoFilt"),
    ("NoGSR_Bandpass", "Bandpass"),
]

for (vname, btag) in variants
    compute_scaffolds_for_variant(vname, btag)
end

println("\nAll scaffold computations complete!")
