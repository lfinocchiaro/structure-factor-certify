#=
Certified bounds vs exact thermodynamic-limit values for two free-fermion chains, tested with Fejér
kernels centred at p0. Produces the CSV tables and figures stored in plots/ (at the repository root).

TFIM  H = Σ_x (-Z_x Z_{x+1} + g X_x)                       (model = :TFIM, parameter θ = g, p0 = 0: ferromagnetic)
  f(x) = Toeplitz determinant (Pfeuty 1970, Barouch–McCoy 1971)
  f̂(p) = m² δ(p) + f̂_reg(p),  m² = (1-g²)^{1/4} for g<1, 0 otherwise

XY    H = Σ_x ((1+γ) X_x X_{x+1} + (1-γ) Y_x Y_{x+1})       (model = :XY, parameter θ = γ, p0 = 1/2: antiferromagnetic in Z)
  f(x) = -G_x G_{-x} for x ≠ 0 (Wick, Lieb–Schultz–Mattis 1961 §4, Barouch–McCoy 1971 at h = 0)
  G_r  = (1/2π) ∫ e^{-ikr} (cos k - iγ sin k)/Λ_k dk,  Λ_k = √(cos²k + γ² sin²k)
  f(x) = 0 for even x ≠ 0, f is even in γ, and f̂ has no Dirac part (gapped for γ ≠ 0, f ∼ 1/x² at γ = 0)

O_N = (f̂, φ̂_N) = Σ_{|x|≤N} φ_N(x) f(x)    exact finite sum = SDP target value

Test function: Fejér kernel centred at ±p0 (same as tf_fejer in src), normalised so that φ̂_N(p0) = 1 for p0 ∈ {0, 1/2},
  φ_N(x) = (1 - |x|/(N+1)) / (N+1) · cos(2π p0 x),   φ̂_N(p) = [K_N(p-p0) + K_N(p+p0)]/2,
  K_N(p) = [sin((N+1)πp) / ((N+1) sin πp)]².

Sections
  1. Exact correlation functions and overlaps (no solver)
  2. Figures of the exact structure factor and overlaps:   figure_panels, figure_overlap
  3. Tables:  table_theoretical (exact O_N),  table_numerical (SDP bounds, Mosek)
  4. Joint plot of the two tables:                        plot_bounds_vs_theory
  5. Bounds on ⟨Z₀Z₁⟩ for the TFIM vs g, several relaxations: figure_correlator_vs_g

Usage (from the repository root, `julia --project=examples`):
  include("examples/chains_1d_plots.jl")
  table_theoretical(:XY); table_numerical(:XY; verbose = true); plot_bounds_vs_theory(:XY)
=#
using StructureFactorCertify
using Dualization, Mosek, MosekTools
using Printf, DelimitedFiles, Plots, LaTeXStrings

const OPTIMIZER   = Dualization.dual_optimizer(Mosek.Optimizer)
const PLOTS_DIR   = joinpath(@__DIR__, "..", "plots")   # figures and CSV tables

timestamp() = Libc.strftime("%d%m_%Hh%M", time())
output_path(name) = joinpath(mkpath(PLOTS_DIR), name)

# Per-model data: parameter name and label, critical point, default sweep, default test-function centre,
# y-range of the joint plot (:auto lets Plots choose), title of the joint plot
const MODELS = Dict(
    :TFIM => (param = "g", label = L"g",      crit = 1.0, range = range(0.05, 2.5, length = 30), p0 = 0.0,
              ylims = (0.0, 1.0), title = "TFIM ferromagnetic"),
    :XY   => (param = "γ", label = L"\gamma", crit = 0.0, range = range(0.0, 1.5, length = 31),  p0 = 0.5,
              ylims = :auto,      title = "XY antiferromagnetic (Z)"),
)

function lattice(model::Symbol, θ)
    model == :TFIM && return Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, θ))
    model == :XY   && return Spin_Lattice_1D(2, XY_1D_Interaction(1.0, θ))
    error("Unsupported model $model")
end


# =====================================================================================================
# 1. Exact correlation functions and overlaps
# =====================================================================================================

# The numerics live in src/exact_chains.jl (exact_correlations, exact_dirac_weight, exact_overlap,
# exact_structure_factor). The wrappers below take a model symbol and parameter θ instead of a lattice.
corr(model::Symbol, θ, R) = exact_correlations(lattice(model, θ), R)
m2(model::Symbol, θ)      = exact_dirac_weight(lattice(model, θ))
tfim_corr(g, R) = corr(:TFIM, g, R)
xy_corr(γ, R)   = corr(:XY, γ, R)
tfim_m2(g)      = m2(:TFIM, g)

"""Σ_x (f(x) - m²) cos(2πpx), |x| ≤ R. Cesàro (Fejér) window removes Gibbs ringing (used at criticality)."""
S_reg(model, θ, p; R = 400, cesaro = false, f = corr(model, θ, R)) = exact_structure_factor(lattice(model, θ), p; R, cesaro, f)

# Fejér kernel in momentum space, φ̂_N(p) = Σ_x φ_N(x) cos(2πpx), with φ_N = tf_fejer(N, p0)
fejer_hat(N, p; p0 = 0.0) = (tf = tf_fejer(N, p0); [sum(phi_at(tf, x) * cospi(2pp * x) for x in -N:N) for pp in p])

overlap(model, θ, N; p0 = MODELS[model].p0, f = corr(model, θ, N)) = exact_overlap(lattice(model, θ), tf_fejer(N, p0); f)

# ---------------- self-checks ----------------
function tfim_selfcheck()
    ref = [(0.5, 4, 0.9457450592), (0.5, 16, 0.9351347149), (1.0, 4, 0.6481440454),
           (1.0, 16, 0.4802440636), (1.5, 4, 0.3789264881), (1.5, 16, 0.1394296768)]
    for (g, N, v) in ref
        o = overlap(:TFIM, g, N; p0 = 0.0)
        @printf("g=%.1f N=%2d  O_N=%.10f  ref=%.10f  %s\n", g, N, o, v, abs(o - v) < 1e-8 ? "ok" : "MISMATCH")
    end
end

"""Closed forms (γ = 0: f(r) = -4/(πr)² for odd r; γ = ±1: f(r) = δ_{r0}), the γ → 0 limit of the
trapezoid rule, and the symmetry γ → -γ."""
function xy_selfcheck(; R = 9)
    ok(b) = b ? "ok" : "MISMATCH"
    xx = [r == 0 ? 1.0 : isodd(r) ? -4 / (π * r)^2 : 0.0 for r in 0:R]
    @printf("γ=0     closed form XX          %s\n", ok(maximum(abs.(xy_corr(0.0, R) .- xx)) < 1e-12))
    ising = [r == 0 ? 1.0 : 0.0 for r in 0:R]
    @printf("γ=±1    δ_{r0}                  %s\n", ok(maximum(abs.(xy_corr(1.0, R) .- ising)) < 1e-12 &&
                                                       maximum(abs.(xy_corr(-1.0, R) .- ising)) < 1e-12))
    @printf("γ=0.01  close to γ=0            %s\n", ok(maximum(abs.(xy_corr(0.01, R) .- xx)) < 0.02))
    @printf("γ=0.6   even in γ               %s\n", ok(xy_corr(0.6, R) ≈ xy_corr(-0.6, R)))
    @printf("γ=0.6   f(1) = %.6f  (ED, 12 sites periodic: -0.12423)\n", xy_corr(0.6, 1)[2])
end


# =====================================================================================================
# 2. Figures of the exact structure factor and overlaps
# =====================================================================================================

# ---------------- Figure 1: three panels ----------------
function tfim_panel(g; N = 8, R = 400)
    p = range(-0.5, 0.5, length = 801)
    crit = isapprox(g, 1.0)
    f = tfim_corr(g, R)
    S = S_reg(:TFIM, g, p; R = R, cesaro = crit, f = f)
    ymax = 3.0 # crit ? 6.0 : 1.3 * maximum(S)
    O = overlap(:TFIM, g, N; p0 = 0.0, f = f)

    pl = plot(p, S; lw = 2, color = :firebrick, xlabel = L"p", ylabel = L"\hat f(p)",
              ylims = (0, ymax), xlims = (-0.5, 0.5), legend = :topright, framestyle = :box,
              label = L"\hat f(p)", #crit ? "Cesàro-summed, R=$R (true: ∼|p|^-3/4)" : L"\hat f_{\rm reg}(p)",
              title = @sprintf("g = %.1f    O_%d = %.4f", g, N, O))
    # Fejér kernel, rescaled to 80% of the panel height (true peak value: 1)
    plot!(pl, p, 2.0 .* fejer_hat(N, p); fillrange = 0, fillalpha = 0.15, lw = 1.5,
          ls = :dash, color = :black, label = "Fejér kernel N=$N (×2)")
    if g < 1   # Dirac peak: arrow, weight m² written next to it
        plot!(pl, [0, 0], [0, 0.92ymax]; arrow = true, lw = 3, color = :firebrick,
        label = @sprintf("m² δ(p),  m² = %.4f", tfim_m2(g)))
    end
    return pl
end

"""XY panel on p ∈ [0, 1], centred on the kernel bump at p0 = 1/2. f̂ has no Dirac part, so
O_N = ∫ f̂ φ̂_N dp exactly (Parseval): the shaded area under f̂·φ̂_N is O_N. No Cesàro window is needed,
since f ∼ 1/x² is absolutely summable even at γ = 0, where f̂(p) = 4|p| on [-1/2, 1/2]."""
function xy_panel(γ; N = 8, R = 400, p0 = 0.5)
    p = range(0, 1, length = 801)
    f = xy_corr(γ, R)
    S = S_reg(:XY, γ, p; R = R, f = f)          # m² = 0: this is the full f̂
    φ̂ = fejer_hat(N, p; p0)
    O = overlap(:XY, γ, N; p0, f = f)

    pl = plot(p, S; lw = 2, color = :firebrick, xlabel = L"p", ylabel = L"\hat f(p)",
              ylims = (0, 2.2), xlims = (0, 1), legend = :topright, framestyle = :box,
              label = L"\hat f(p)", title = @sprintf("γ = %.1f    O_%d = %.4f", γ, N, O))
    plot!(pl, p, φ̂; fillrange = 0, fillalpha = 0.10, lw = 1.5, ls = :dash, color = :black,
          label = "Fejér kernel N=$N, p0=$p0")
    plot!(pl, p, S .* φ̂; fillrange = 0, fillalpha = 0.35, lw = 0, color = :firebrick,
          label = L"\hat f\,\hat\varphi_N\ \ (\mathrm{area} = O_N)")
    return pl
end

function figure_panels(model = :TFIM; N = 8, vals = model == :XY ? [0.0, 0.5, 1.5] : [0.9, 1.0, 2.0])
    panel = model == :TFIM ? tfim_panel : model == :XY ? xy_panel : error("Unknown model $model")
    plt = plot((panel(θ; N = N) for θ in vals)...;
               layout = (1, length(vals)), size = (530 * length(vals), 480), margin = 6Plots.mm)
    savefig(plt, output_path("$(lowercase(string(model)))_panels_fejer_$(timestamp()).png"))
    return plt
end

# ---------------- Figure 2: target overlaps vs parameter ----------------
"""O_N(θ) for several N. For N → ∞ the overlap tends to the Dirac weight at p0 (φ̂_N(p0) = 1 but ∫ φ̂_N → 0):
m² for the TFIM at p0 = 0, and 0 for the XY model, which is not drawn."""
function figure_overlap(model = :TFIM; Ns = [2, 4, 8, 16, 32, 64], θs = range(MODELS[model].range[1], MODELS[model].range[end], length = 120),
                        p0 = MODELS[model].p0)
    info = MODELS[model]
    Nmax = maximum(Ns)
    F = [corr(model, θ, Nmax) for θ in θs]           # one correlation sweep per θ, reused for all N
    plt = plot(xlabel = info.label, ylabel = L"O_N = \sum_x \varphi_N(x) f(x)",
               title = "$model: SDP target values (Fejér, p0 = $p0)", size = (700, 500), framestyle = :box)
    cols = palette(:viridis, length(Ns) + 1)
    for (i, N) in enumerate(Ns)
        plot!(plt, θs, [overlap(model, θ, N; p0, f = F[j]) for (j, θ) in enumerate(θs)];
              lw = 2, color = cols[i], label = "N = $N")
    end
    if model == :TFIM && iszero(p0)
        plot!(plt, θs, tfim_m2.(θs); lw = 2.5, ls = :dash, color = :black, label = L"N\to\infty:\ (1-g^2)^{1/4}")
    end
    savefig(plt, output_path("$(lowercase(string(model)))_overlap_p0_$(p0)_$(timestamp()).png"))
    return plt
end


# =====================================================================================================
# 3. Tables: exact overlaps and certified bounds
# =====================================================================================================

"""Store the exact overlaps (f̂, φ̂_N) = Σ_{|x|≤N} φ_N(x) f(x) for several N and θ in
plots/table_theoretical_<model>_<date>.csv. Header: `θ/N,N1,N2,...`."""
function table_theoretical(model = :TFIM; N_list = [1,2,3], θ_list = MODELS[model].range, p0 = MODELS[model].p0)
    filename = output_path("table_theoretical_$(model)_$(timestamp()).csv")

    F = [corr(model, θ, maximum(N_list)) for θ in θ_list]       # one correlation sweep per θ, reused for all N
    data = [overlap(model, θ, N; p0, f = F[j]) for (j, θ) in enumerate(θ_list), N in N_list]

    open(filename, "w") do io
        writedlm(io, permutedims(["$(MODELS[model].param)/N"; string.(N_list)]), ',')   # header: corner label + N values
        writedlm(io, hcat(θ_list, data), ',')                                          # each row: θ, then data[i, :]
    end
    return filename
end

"""Certified bounds on (f̂, φ̂_N) for several N and θ, stored in plots/table_numerical_<model>_<date>.csv.
Columns: θ, lo_N1, hi_N1, lo_N2, hi_N2, ..."""
function table_numerical(model = :TFIM; N_list = [1,2,3], θ_list = MODELS[model].range, p0 = MODELS[model].p0,
                         L = 7, verbose = false, optimizer = OPTIMIZER)
    @assert L ≥ 2maximum(N_list) + 1 "table_numerical: L must be at least 2N+1 for the largest N in N_list"
    filename = output_path("table_numerical_$(model)_$(timestamp()).csv")

    data = zeros(Float64, length(θ_list), length(N_list), 2)
    for (j, θ) in enumerate(θ_list)
        system = lattice(model, θ)
        for (i, N) in enumerate(N_list)
            verbose && print("Computing bounds for θ=$(θ), N=$(N), L=$(L)...")
            tf = tf_fejer(N, p0)
            (lo, hi), _ = compute_bounds(StructureFactor_Problem(system, tf),
                                    Relaxation(L = L, d = 2, use_translation=true); optimizer, verbose = false)
            data[j, i, 1] = lo
            data[j, i, 2] = hi
            verbose && println("done: [$(lo), $(hi)]")
        end
    end

    header = ["θ"; vec(["$(s)_$N" for s in ("lo", "hi"), N in N_list])]
    body   = hcat(θ_list, reshape(permutedims(data, (1, 3, 2)), length(θ_list), :))
    open(filename, "w") do io
        writedlm(io, permutedims(header), ',')
        writedlm(io, body, ',')
    end
    return filename
end


# =====================================================================================================
# 4. Joint plot of the certified bounds and the exact overlaps (reads the CSV tables, no solver)
# =====================================================================================================

"""
    read_table(filename)

Read a table written by `table_theoretical` or `table_numerical`.
- theoretical (header `θ/N,2,3,...`):            returns (; g, N, values), values[i, j] = O_{N[j]}(g[i])
- numerical   (header `θ,lo_2,hi_2,lo_3,...`):   returns (; g, N, lo, hi)
The first column `g` holds the model parameter θ (g for the TFIM, γ for the XY model).
"""
function read_table(filename)
    M, hdr = readdlm(filename, ',', Float64; header = true)
    cols = vec(hdr)[2:end]
    g = M[:, 1]
    if all(c -> startswith(c, "lo_") || startswith(c, "hi_"), cols)
        N = parse.(Int, [c[4:end] for c in cols[1:2:end]])
        return (; g, N, lo = M[:, 2:2:end], hi = M[:, 3:2:end])
    else
        return (; g, N = parse.(Int, cols), values = M[:, 2:end])
    end
end

"""
    latest_table(prefix; dir = PLOTS_DIR)

Most recently modified `<prefix>*.csv` file in `dir` that contains data (header-only files,
e.g. from an interrupted run, are skipped). Uses the modification time rather than the
date in the filename, since `%d%m_%Hh%M` does not sort chronologically.
Use e.g. `latest_table("table_theoretical_XY_p0_0.5_")` for one model and test-function centre.
"""
function latest_table(prefix; dir = PLOTS_DIR)
    files = [joinpath(dir, f) for f in readdir(dir) if startswith(f, prefix) && endswith(f, ".csv")]
    filter!(f -> countlines(f) ≥ 2, files)
    isempty(files) && error("latest_table: no non-empty $(prefix)*.csv file in $dir")
    return files[argmax(mtime.(files))]
end

"""
    plot_bounds_vs_theory(model = :TFIM, theo_file = :latest, num_file = :latest; Ns = nothing, save = true, L = 7)

One shaded band [lo, hi] per N from the numerical table, with the exact overlap O_N(θ)
from the theoretical table as a dashed line of the same colour.
Pass `:latest` (default) to use the most recent table of each kind for `model` in plots/,
or a path to use a specific file. `Ns` selects which N to plot (default: all N of the numerical table).
`L` is only used in the title.
"""
function plot_bounds_vs_theory(model = :TFIM, theo_file = :latest, num_file = :latest; Ns = nothing, save = true, L = 7)
    info = MODELS[model]
    theo_file === :latest && (theo_file = latest_table("table_theoretical_$(model)_"))
    num_file  === :latest && (num_file  = latest_table("table_numerical_$(model)_"))
    println("Theoretical table: $theo_file\nNumerical table:   $num_file")

    th, nu = read_table(theo_file), read_table(num_file)
    Ns = something(Ns, nu.N)
    cols = palette(:viridis, length(Ns) + 1)

    plt = plot(xlabel = info.label, ylabel = L"(\hat f, \hat\varphi_N)",
               title = "$(info.title): certified bounds vs theoretical L=$L", legend = :topright,
               framestyle = :box, size = (700, 500), dpi = 150)
    for (k, N) in enumerate(Ns)
        j = findfirst(==(N), nu.N)
        if j !== nothing
            plot!(plt, nu.g, nu.lo[:, j]; fillrange = nu.hi[:, j], fillalpha = 0.25, lw = 1,
                  color = cols[k], marker = :circle, ms = 2, msw = 0, label = "bounds, N = $N")
            plot!(plt, nu.g, nu.hi[:, j]; lw = 1, color = cols[k], marker = :circle, ms = 2, msw = 0, label = "")
        end
        j = findfirst(==(N), th.N)
        j === nothing || plot!(plt, th.g, th.values[:, j]; lw = 2, ls = :dash, color = cols[k],
                               label = "exact, N = $N")
    end
    vline!(plt, [info.crit]; color = :gray, ls = :dot, label = "")   # critical point
    plot!(plt; ylims = info.ylims)   # the attribute accepts :auto, ylims!(plt, ::Symbol) does not

    save && savefig(plt, output_path("$(lowercase(string(model)))_bounds_vs_exact_$(timestamp()).png"))
    return plt
end


# =====================================================================================================
# 5. Certified bounds on ⟨Z₀Z₁⟩ for the TFIM vs g, for several relaxation levels, vs the exact value
# =====================================================================================================

"""`Ld_list` holds tuples (L, d, rdm_size); rdm_size = 0 means no RDM constraints."""
function figure_correlator_vs_g(; g_list = range(start = 0.1, stop = 2.0, length = 20),
                                Ld_list = ((3, 2, 0), (5, 2, 0), (7, 2, 0), (7, 2, 4)),
                                kms = CommutatorKMS(),   # or AnticommutatorKMS()
                                optimizer = OPTIMIZER)
    # Compute the bounds
    all_results = zeros(Float64, length(g_list), length(Ld_list), 2)  # g, (L,d,rdm_size), (lo, hi)
    for (ig, g) in enumerate(g_list)
        prob = StructureFactor_Problem(Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, g)), tf_single_correlator(1))
        for (ild, (L, d, rdm_size)) in enumerate(Ld_list)
            t = @elapsed (lo, hi), _ = compute_bounds(prob, Relaxation(; L, d, kms, rdm_size); optimizer, verbose = false)
            @printf("g = %.2f, L = %d, d = %d, rdm_size = %d : [%.7f, %.7f]  %.1f s\n", g, L, d, rdm_size, lo, hi, t)
            all_results[ig, ild, :] = [lo, hi]
        end
    end

    # Plot: one shaded band [lo, hi] per relaxation level, plus the exact value
    colors = palette(:viridis, length(Ld_list) + 1)
    p = plot(xlabel = "g", ylabel = raw"$\langle Z_0 Z_1 \rangle$", title = raw"TFIM (J = 1): certified bounds on $\langle Z_0 Z_1 \rangle$",
             legend = :bottomright, framestyle = :box, size = (450, 450), dpi = 150)
    for (ild, (L, d, rdm_size)) in enumerate(Ld_list)
        lo, hi = all_results[:, ild, 1], all_results[:, ild, 2]
        plot!(p, g_list, lo; fillrange = hi, fillalpha = 0.25, color = colors[ild],
              lw = 1.5, marker = :circle, ms = 3, msw = 0, label = "L = $L, d = $d" * (rdm_size > 0 ? ", RDM size $rdm_size" : ""))
        plot!(p, g_list, hi; color = colors[ild], lw = 1.5, marker = :circle, ms = 3, msw = 0, label = "")
    end
    g_fine = range(first(g_list), last(g_list), length = 300)
    plot!(p, g_fine, tfim_nn_correlator_numerical.(g_fine); color = :black, ls = :dash, lw = 2,
          label = "exact (thermodynamic limit)")
    vline!(p, [1.0]; color = :gray, ls = :dot, label = "")   # critical point g = 1

    filename = output_path("tfim_correlator_vs_g_$(timestamp()).png")
    savefig(p, filename)
    println("Figure saved to $filename")
    return p
end


# =====================================================================================================
# Usage
# =====================================================================================================

# tfim_selfcheck()
# xy_selfcheck()
# figure_panels(:TFIM; N = 12, vals = [0.8, 1.0, 2.5])
# figure_panels(:XY; N = 8)
# figure_overlap(:TFIM)
# figure_overlap(:XY)

# table_theoretical(:TFIM);  table_numerical(:TFIM; verbose = true);  plot_bounds_vs_theory(:TFIM)
# table_theoretical(:XY);    table_numerical(:XY;   verbose = true);  plot_bounds_vs_theory(:XY)
# plot_bounds_vs_theory(:TFIM, joinpath(PLOTS_DIR, "table_theoretical_0810_17h37.csv"))   # specific theoretical, latest numerical
# figure_correlator_vs_g()
