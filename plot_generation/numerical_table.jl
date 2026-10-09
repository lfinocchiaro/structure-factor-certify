# Certified bounds on the structure factor of 1D chains, tested with Fejér kernels centred at p0:
#   TFIM (θ = g): ferromagnetic correlations, p0 = 0
#   XY   (θ = γ): antiferromagnetic Z-correlations, p0 = 1/2

using StructureFactorCertify
using Dualization, Mosek, MosekTools
using Printf, DelimitedFiles

optimizer = Dualization.dual_optimizer(Mosek.Optimizer)

function lattice(model::Symbol, θ)
    model == :TFIM && return Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, θ))
    model == :XY   && return Spin_Lattice_1D(2, XY_1D_Interaction(1.0, θ))
    error("Unsupported model $model")
end

function table_numerical(model = :TFIM; N_list = [1,2,3],
                         θ_list = model == :XY ? range(0.0, 1.5, length = 31) : range(0.05, 2.5, length = 30),
                         p0 = model == :XY ? 0.5 : 0.0, L = 7, verbose = false)
    @assert L ≥ 2maximum(N_list) + 1 "table_numerical: L must be at least 2N+1 for the largest N in N_list"
    filename = joinpath(@__DIR__, "table_numerical_$(model)_p0_$(p0)_$(Libc.strftime("%d%m_%Hh%M", time())).csv")

    data = zeros(Float64, length(θ_list), length(N_list), 2)
    for (j, θ) in enumerate(θ_list)
        system = lattice(model, θ)
        for (i, N) in enumerate(N_list)
            verbose && print("Computing bounds for θ=$(θ), N=$(N), L=$(L)...")
            tf = tf_fejer(N, p0)
            (lo, hi), _ = compute_bounds(StructureFactor_Problem(system, tf), Relaxation(L = L, d = 2); optimizer, verbose = false)
            data[j, i, 1] = lo
            data[j, i, 2] = hi
            verbose && println("done: [$(lo), $(hi)]")
        end
    end

    # Columns: θ, lo_N1, hi_N1, lo_N2, hi_N2, ...
    header = ["θ"; vec(["$(s)_$N" for s in ("lo", "hi"), N in N_list])]
    body   = hcat(θ_list, reshape(permutedims(data, (1, 3, 2)), length(θ_list), :))
    open(filename, "w") do io
        writedlm(io, permutedims(header), ',')
        writedlm(io, body, ',')
    end
    return filename
end

# table_numerical(:TFIM; verbose = true)
# table_numerical(:XY;   verbose = true)
