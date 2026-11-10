# Computation of the convolution f ⋆ ϕₙ, i.e. a smooth version of f where ϕₙ is the n-th Fejér kernel
# 3-panel plot: theoretical structure factor (with δ peak), N=3, N=4
# The last two plots have certified bounds for different (L, d)


using StructureFactorCertify
using Dualization, Mosek, MosekTools
using Printf, DelimitedFiles, Plots, LaTeXStrings

const PLOTS_DIR = joinpath(@__DIR__, "..", "plots")   # figures and CSV tables

optimizer = Dualization.dual_optimizer(Mosek.Optimizer)

g = 0.5
system = Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, g))

""" Compute the list of bounds on (f ⋆ ϕ_N)(p) for p in p_list with given relaxation parameters.
This is equivalent to computing the overlap (f, ϕᵖ_N) for the shifted kernel ϕᵖ_N.
Returns a length(p_list) × 2 matrix whose rows are [lo, hi]. """
function convolution_bounds(system::Spin_Lattice_1D, N::Int, p_list::AbstractVector{<:Real}, relaxation::Relaxation;
                            optimizer = optimizer, verbose = false)
    res = zeros(Float64, length(p_list), 2)
    for (ip, p) in enumerate(p_list)
        ϕ = tf_fejer(N, p)
        (lo, hi), _ = compute_bounds(StructureFactor_Problem(system, ϕ), relaxation; optimizer, verbose)
        res[ip, :] .= (lo, hi)
    end
    return res
end

""" Return the list of theoretical values for (f ⋆ ϕ_N)(p) for p in p_list.
This is the same as the overlaps (f, ϕᵖ_N) = Σ_{|x|≤N} ϕᵖ_N(x) f(x), with the exact
thermodynamic-limit correlations f(x) = ⟨Z_0 Z_x⟩ (TFIM with J > 0, or XY chain). """
function convolution_th(system::Spin_Lattice_1D, N::Int, p_list::AbstractVector{<:Real})
    f = exact_correlations(system, N)          # f(0), …, f(N), computed once for all p
    return [exact_overlap(system, tf_fejer(N, p); f) for p in p_list]
end


# =====================================================================================================
# Table: exact values and certified bounds on (f ⋆ ϕ_N)(p), stored in a single CSV file
# =====================================================================================================

# (model name, J, parameter θ) of a chain, and back
function model_params(system::Spin_Lattice_1D)
    int = system.interaction
    int isa TFIM_1D_Interaction && return ("TFIM", int.J, int.g)
    int isa XY_1D_Interaction   && return ("XY", int.J, int.γ)
    error("sf_sweep: unsupported interaction $int")
end

function system_from_params(model, J, θ)
    model == "TFIM" && return Spin_Lattice_1D(2, TFIM_1D_Interaction(J, θ))
    model == "XY"   && return Spin_Lattice_1D(2, XY_1D_Interaction(J, θ))
    error("sf_sweep: unsupported model $model")
end

"""
    sf_sweep_table(system; N_list = [3, 4], p_list = range(0, 0.5, length = 21), Ld_list = [(7, 2), (9, 2)],
                   relaxation = (L, d) -> Relaxation(; L, d, use_translation = true), optimizer, verbose = true)

Compute the exact value and the certified bounds of (f ⋆ ϕ_N)(p) for every N in `N_list`, p in `p_list`
and relaxation level (L, d) in `Ld_list`, and store them in plots/sf_sweep_<model>_<θ>_<date>.csv.
Levels with L < 2N+1 are skipped for that N (the test function does not fit in the window).
Since f̂ and the Fejér kernel are even, p ∈ [0, 1/2] is enough: `plot_sf_sweep` mirrors the data to p < 0.

One row per (N, L, d, p), with columns `model,J,theta,N,L,d,p,th,lo,hi`. Rows are written as soon as
they are computed, so an interrupted run leaves a usable (partial) file. Returns the file name.
"""
function sf_sweep_table(system::Spin_Lattice_1D; N_list = [3, 4], p_list = range(0, 0.5, length = 21),
                        Ld_list = [(7, 2), (9, 2)],
                        relaxation = (L, d) -> Relaxation(; L, d, use_translation = true),
                        optimizer = optimizer, verbose = true)
    model, J, θ = model_params(system)
    p_list = collect(p_list)
    filename = joinpath(mkpath(PLOTS_DIR), "sf_sweep_$(model)_$(θ)_$(Libc.strftime("%d%m_%Hh%M", time())).csv")

    open(filename, "w") do io
        println(io, "model,J,theta,N,L,d,p,th,lo,hi")
        for N in N_list
            th = convolution_th(system, N, p_list)
            for (L, d) in Ld_list
                if L < 2N + 1
                    @warn "sf_sweep_table: skipping (L, d) = ($L, $d) for N = $N, need L ≥ $(2N + 1)"
                    continue
                end
                relax = relaxation(L, d)
                for (ip, p) in enumerate(p_list)
                    t = @elapsed bounds = convolution_bounds(system, N, [p], relax; optimizer)
                    lo, hi = bounds[1, 1], bounds[1, 2]
                    verbose && @printf("N = %d, L = %d, d = %d, p = %.4f : [%.7f, %.7f]  exact %.7f  %.1f s\n",
                                       N, L, d, p, lo, hi, th[ip], t)
                    println(io, join((model, J, θ, N, L, d, p, th[ip], lo, hi), ','))
                    flush(io)
                end
            end
        end
    end
    return filename
end


# =====================================================================================================
# Figure: exact structure factor, then one panel per N with the smoothed f ⋆ ϕ_N and its certified bounds
# =====================================================================================================

"Most recently modified sf_sweep_*.csv file in `dir` that contains data."
function latest_sf_sweep(; dir = PLOTS_DIR)
    files = [joinpath(dir, f) for f in readdir(dir) if startswith(f, "sf_sweep_") && endswith(f, ".csv")]
    filter!(f -> countlines(f) ≥ 2, files)
    isempty(files) && error("latest_sf_sweep: no non-empty sf_sweep_*.csv file in $dir")
    return files[argmax(mtime.(files))]
end

# Extend data given on p ≥ 0 to -p by parity (the point p = 0 is not duplicated)
function mirror(p, v)
    idx = [i for i in reverse(eachindex(p)) if p[i] > 0]
    return [-p[idx]; p], [v[idx]; v]
end

# Exact structure factor f̂(p) on [-1/2, 1/2]: regular part, plus an arrow for the Dirac peak m² δ(p) if any
function panel_exact(system, title; R = 400, ylims = (0.0, 1.0))
    model, J, θ = model_params(system)
    p  = range(-0.5, 0.5, length = 801)
    m² = exact_dirac_weight(system)
    critical = model == "TFIM" ? isapprox(abs(θ), J) : iszero(θ)
    S  = exact_structure_factor(system, p; R, cesaro = critical)   # Cesàro window removes the ringing at criticality
    ymax = ylims[2]

    pl = plot(p, S; lw = 2, color = :firebrick, xlabel = L"p", ylabel = L"\hat f(p)", title,
              xlims = (-0.5, 0.5), ylims, legend = :topright, framestyle = :box,
              label = m² > 0 ? L"\hat f_{\rm reg}(p)" : L"\hat f(p)")
    if m² > 0
        plot!(pl, [0, 0], [0, 0.92ymax]; arrow = true, lw = 3, color = :firebrick,
              label = @sprintf("m² δ(p),  m² = %.4f", m²))
    end
    return pl
end

"""
    plot_sf_sweep(filename = :latest; ylims = (0.0, 1.0), save = true)

Read a table written by `sf_sweep_table` and draw one row of panels: the exact structure factor f̂
(with its Dirac peak), then for each N of the table the exact smoothed version (f ⋆ ϕ_N)(p) as a dashed
line, with one shaded band [lo, hi] of certified bounds per relaxation level (L, d).
`:latest` uses the most recent sf_sweep_*.csv file of plots/. The figure is saved in plots/ if `save`.
All panels share the y-range `ylims` (the smoothed values lie in [0, 1], the regular part of f̂ may exceed 1).
"""
function plot_sf_sweep(filename = :latest; ylims = (0.0, 1.0), save = true)
    filename === :latest && (filename = latest_sf_sweep())
    println("Table: $filename")
    M, _ = readdlm(filename, ','; header = true)        # columns: model, J, theta, N, L, d, p, th, lo, hi
    model, J, θ = string(M[1, 1]), Float64(M[1, 2]), Float64(M[1, 3])
    system = system_from_params(model, J, θ)
    Ns, Ls, ds = Int.(M[:, 4]), Int.(M[:, 5]), Int.(M[:, 6])
    p, th, lo, hi = (Float64.(M[:, k]) for k in 7:10)

    N_list  = unique(Ns)
    Ld_list = unique(collect(zip(Ls, ds)))               # same colour for a level (L, d) in all panels
    colors  = palette(:viridis, length(Ld_list) + 1)
    param   = model == "TFIM" ? "g" : "γ"

    panels = Any[panel_exact(system, "$model, $param = $θ: exact structure factor"; ylims)]
    for N in N_list
        pl = plot(xlabel = L"p", ylabel = L"(\hat f \star \hat\varphi_N)(p)", title = "N = $N",
                  xlims = (-0.5, 0.5), ylims = ylims, legend = :topright, framestyle = :box)
        for (k, (L, d)) in enumerate(Ld_list)
            rows = findall(i -> Ns[i] == N && Ls[i] == L && ds[i] == d, eachindex(Ns))
            isempty(rows) && continue
            rows = rows[sortperm(p[rows])]
            pm, lom = mirror(p[rows], lo[rows])
            _,  him = mirror(p[rows], hi[rows])
            plot!(pl, pm, lom; fillrange = him, fillalpha = 0.25, lw = 1, color = colors[k],
                  marker = :circle, ms = 2, msw = 0, label = "bounds, L = $L, d = $d")
            plot!(pl, pm, him; lw = 1, color = colors[k], marker = :circle, ms = 2, msw = 0, label = "")
        end
        # exact values: identical for all levels (L, d), take each p once
        rows = findall(==(N), Ns)
        rows = rows[unique(i -> p[rows[i]], eachindex(rows))]
        rows = rows[sortperm(p[rows])]
        pm, thm = mirror(p[rows], th[rows])
        plot!(pl, pm, thm; lw = 2, ls = :dash, color = :black, label = "exact")
        push!(panels, pl)
    end

    plt = plot(panels...; layout = (1, length(panels)), size = (530 * length(panels), 480), margin = 6Plots.mm, dpi = 150)
    if save
        figname = joinpath(mkpath(PLOTS_DIR), "sf_sweep_$(model)_$(θ)_$(Libc.strftime("%d%m_%Hh%M", time())).png")
        savefig(plt, figname)
        println("Figure saved to $figname")
    end
    return plt
end

# sf_sweep_table(system)      # long: one SDP per (N, L, d, p)
# plot_sf_sweep()             # most recent table of plots/


