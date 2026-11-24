"""
Compute sliding window scaffolds for HCP 1200-subject dataset using Julia, on the 300 CORTICAL regions only.
Optimized to save frequency as Int16 and persistence as Float32.

Output: NoGSR_LowPassOnly/scaffolds/windows_cortex/<width>/{freq,pers}/<sub>_s01.npy
Resumable: existing outputs are skipped.

Usage:
  julia -t 8 scripts/02_scaffolds_windows_1200.jl
  julia -t 8 scripts/02_scaffolds_windows_1200.jl --test   # 1 subject into ./windows_test
"""

using DelimitedFiles
using Statistics
using Ripserer
using ProgressMeter
using NPZ

const N_CORTICAL = 300

function compute_scaffolds_windows()
    test = "--test" in ARGS
    data_dir_ts = joinpath(get(ENV, "TOPOFINGER_HCP_ROOT", "./data/HCP/1200subjects/schaefer300"), "NoGSR_LowPassOnly", "ts")
    out_dir_base = test ? joinpath(pwd(), "windows_test") : joinpath(get(ENV, "TOPOFINGER_HCP_ROOT", "./data/HCP/1200subjects/schaefer300"), "NoGSR_LowPassOnly", "scaffolds", "windows_cortex")
    
    widths = [1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50]
    shifts = [100,  100,  100, 100, 100, 120, 140, 160, 180, 200, 220, 230]

    lr = "LR"
    atlas = "Schaefer300"
    gsr = "NoGSR"
    bandpass_tag = "LowPassOnly"

    files = readdir(data_dir_ts)
    subjects = sort(collect(Set(split(f, '_')[1] for f in files)))
    test && (subjects = subjects[1:1])
    n = length(subjects)

    for (i, width) in enumerate(widths)
        shift = shifts[i]
        println("\n========================================")
        println("Processing window width: $width (shift: $shift)")
        println("========================================")
        
        out_dir_width = joinpath(out_dir_base, string(width))
        out_dir_freq = joinpath(out_dir_width, "freq")
        out_dir_pers = joinpath(out_dir_width, "pers")
        
        mkpath(out_dir_freq)
        mkpath(out_dir_pers)
        
        p = ProgressMeter.Progress(n, desc="[Width $width] ")

        Threads.@threads for sub in subjects
            try
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

                out_freq_path = joinpath(out_dir_freq, "$(sub)_s01.npy")
                out_pers_path = joinpath(out_dir_pers, "$(sub)_s01.npy")
                
                # If already computed, skip
                if isfile(out_freq_path) && isfile(out_pers_path)
                    ProgressMeter.next!(p)
                    continue
                end

                # Load both sessions into memory
                ts_sessions = []
                for fname in sess_files
                    file_path = joinpath(data_dir_ts, fname)
                    data = npzread(file_path)
                    ts = data["timeseries"]'[:, 1:N_CORTICAL]   # (time, 300), cortical regions only
                    push!(ts_sessions, ts)
                end
                
                # fld, not div: a run shorter than the window gives no_win = 0, not 1
                no_win1 = fld(size(ts_sessions[1], 1) - width, shift) + 1
                no_win2 = fld(size(ts_sessions[2], 1) - width, shift) + 1
                no_win = min(no_win1, no_win2)
                
                if no_win <= 0
                    ProgressMeter.next!(p)
                    continue
                end
                
                total_windows = no_win * 2
                n_regions = size(ts_sessions[1], 2)
                
                f_scaffolds = zeros(Int16, n_regions, n_regions, total_windows)
                p_scaffolds = zeros(Float32, n_regions, n_regions, total_windows)
                
                window_idx = 1
                for n_sess in (1, 2)
                    ts = ts_sessions[n_sess]
                    for t in 1:no_win
                        start_idx = (t - 1) * shift + 1
                        end_idx = start_idx + width - 1
                        ts_win = ts[start_idx:end_idx, :]
                        
                        pms = cor(ts_win)
                        
                        dist_corrs = 1 .- pms
                        dist_corrs = (dist_corrs .+ transpose(dist_corrs)) / 2
                        
                        # Compute Ripserer on the distance matrix using :involuted for massive speedup
                        d = ripserer(dist_corrs; reps=1, alg=:involuted, verbose=false)
                        
                        # Compute scaffolds
                        for l in 1:length(d[2])
                            g = d[2][l]
                            pers = persistence(g)
                            for rep in representative(g)
                                u, v = vertices(rep)
                                p_scaffolds[u, v, window_idx] += Float32(pers)
                                f_scaffolds[u, v, window_idx] += Int16(1)
                                p_scaffolds[v, u, window_idx] += Float32(pers)
                                f_scaffolds[v, u, window_idx] += Int16(1)
                            end
                        end
                        window_idx += 1
                    end
                end
                
                # Write to disk as .npy files
                npzwrite(out_freq_path, f_scaffolds)
                npzwrite(out_pers_path, p_scaffolds)
                
                ProgressMeter.next!(p)
            catch e
                println("\nError processing sub $sub: ", typeof(e))
                ProgressMeter.next!(p)
            end
        end
    end
    println("Done computing scaffolds for all widths!")
end

compute_scaffolds_windows()
