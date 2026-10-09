# Joint plot of the certified bounds (table_numerical, numerical_table.jl) and the exact
# thermodynamic-limit overlaps (table_theoretical, theoretical_plots.jl), read from the CSV tables.
# No solver needed: only the CSV files in this folder are read.

using DelimitedFiles, Plots, LaTeXStrings

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
    latest_table(prefix; dir = @__DIR__)

Most recently modified `<prefix>*.csv` file in `dir` that contains data (header-only files,
e.g. from an interrupted run, are skipped). Uses the modification time rather than the
date in the filename, since `%d%m_%Hh%M` does not sort chronologically.
Use e.g. `latest_table("table_theoretical_XY_p0_0.5_")` for one model and test-function centre.
"""
function latest_table(prefix; dir = @__DIR__)
    files = [joinpath(dir, f) for f in readdir(dir) if startswith(f, prefix) && endswith(f, ".csv")]
    filter!(f -> countlines(f) ≥ 2, files)
    isempty(files) && error("latest_table: no non-empty $(prefix)*.csv file in $dir")
    return files[argmax(mtime.(files))]
end

# Per-model plot settings: parameter label, critical point, y-range (:auto lets Plots choose), title
const PLOT_INFO = Dict(
    :TFIM => (label = L"g",      crit = 1.0, ylims = (0.0, 1.0), title = "TFIM ferromagnetic"),
    :XY   => (label = L"\gamma", crit = 0.0, ylims = :auto,      title = "XY antiferromagnetic (Z)"),
)

"""
    plot_bounds_vs_theory(model = :TFIM, theo_file = :latest, num_file = :latest; p0 = nothing, Ns = nothing, save = true)

One shaded band [lo, hi] per N from the numerical table, with the exact overlap O_N(θ)
from the theoretical table as a dashed line of the same colour.
Pass `:latest` (default) to use the most recent table of each kind for `model` and `p0` in this folder
(default p0: 0 for the TFIM, 1/2 for the XY model), or a path to use a specific file.
`Ns` selects which N to plot (default: all N of the numerical table).
"""
function plot_bounds_vs_theory(model = :TFIM, theo_file = :latest, num_file = :latest; p0 = nothing, Ns = nothing, save = true)
    info = PLOT_INFO[model]
    p0 = something(p0, model == :XY ? 0.5 : 0.0)
    theo_file === :latest && (theo_file = latest_table("table_theoretical_$(model)_p0_$(p0)_"))
    num_file  === :latest && (num_file  = latest_table("table_numerical_$(model)_p0_$(p0)_"))
    println("Theoretical table: $theo_file\nNumerical table:   $num_file")

    th, nu = read_table(theo_file), read_table(num_file)
    Ns = something(Ns, nu.N)
    cols = palette(:viridis, length(Ns) + 1)

    plt = plot(xlabel = info.label, ylabel = L"(\hat f, \hat\varphi_N)",
               title = "$(info.title): certified bounds vs theoretical, p0 = $p0", legend = :topright,
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

    if save
        plots_dir = mkpath(joinpath(@__DIR__, "..", "plots"))
        savefig(plt, joinpath(plots_dir, "$(lowercase(string(model)))_bounds_vs_exact_p0_$(p0)_$(Libc.strftime("%d%m_%Hh%M", time())).png"))
    end
    return plt
end

# plot_bounds_vs_theory(:TFIM)                                                       # most recent TFIM tables
# plot_bounds_vs_theory(:XY)                                                         # most recent XY tables (p0 = 1/2)
# plot_bounds_vs_theory(:TFIM, "plot_generation/table_theoretical_0810_17h37.csv")   # specific theoretical, latest numerical
