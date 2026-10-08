# Certified bounds on the structure factor of the TFIM chain
# Probing ferromagnetic correlations by testing the structure factor at p = 0

using StructureFactorCertify
using Dualization, Mosek, MosekTools
using Printf, DelimitedFiles

optimizer = Dualization.dual_optimizer(Mosek.Optimizer)

function table_numerical(; N_list=[1,2,3], g_list=range(0.05, 2.5, length=30), L=7, verbose=false)
    @assert L ≥ maximum(N_list) "bounds_sf_plots.jl: L must be at least the maximum N in N_list"
    filename = joinpath(@__DIR__, "table_numerical_$(Libc.strftime("%d%m_%Hh%M", time())).csv")

    data = zeros(Float64, length(g_list), length(N_list), 2)
    for (j, g) in enumerate(g_list)
        system = Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, g))
        for (i, N) in enumerate(N_list)
            verbose && print("Computing bounds for g=$(g), N=$(N), L=$(L)...")
            tf = tf_fejer(N)
            (lo, hi), _ = compute_bounds(StructureFactor_Problem(system, tf), Relaxation(L = L, d = 2); optimizer, verbose = false)
            data[j, i, 1] = lo
            data[j, i, 2] = hi
            verbose && println("done: [$(lo), $(hi)]")
        end
    end

    # Columns: g, lo_N1, hi_N1, lo_N2, hi_N2, ...
    header = ["g"; vec(["$(s)_$N" for s in ("lo", "hi"), N in N_list])]
    body   = hcat(g_list, reshape(permutedims(data, (1, 3, 2)), length(g_list), :))
    open(filename, "w") do io
        writedlm(io, permutedims(header), ',')
        writedlm(io, body, ',')
    end
end

