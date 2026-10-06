# Certified bounds on ⟨Z₀Z₁⟩ for the TFIM as a function of the transverse field g,
# for several relaxation levels, compared with the exact thermodynamic-limit value

using StructureFactorCertify
using Dualization, Mosek, MosekTools
using Printf
using Plots

optimizer = Dualization.dual_optimizer(Mosek.Optimizer)

g_list  = range(start = 0.1, stop = 2.0, length = 20)
Ld_list = ((3, 2, 0), (5, 2, 0), (7, 2, 0), (7, 2, 4))   # (L, d, rdm_size), rdm_size = 0: no RDM constraints
kms     = CommutatorKMS()   # or AnticommutatorKMS()

## Compute the bounds
all_results = zeros(Float64, length(g_list), length(Ld_list), 2)  # g, (L,d,rdm_size), (lo, hi)
for (ig, g) in enumerate(g_list)
    prob = StructureFactor_Problem(Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, g)), tf_single_correlator(1))
    for (ild, (L, d, rdm_size)) in enumerate(Ld_list)
        t = @elapsed (lo, hi), _ = compute_bounds(prob, Relaxation(; L, d, kms, rdm_size); optimizer, verbose = false)
        @printf("g = %.2f, L = %d, d = %d, rdm_size = %d : [%.7f, %.7f]  %.1f s\n", g, L, d, rdm_size, lo, hi, t)
        all_results[ig, ild, :] = [lo, hi]
    end
end

## Plot: one shaded band [lo, hi] per relaxation level, plus the exact value
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
display(p)

## Save the figure in plots/ (at the root of the repository)
plots_dir = mkpath(joinpath(@__DIR__, "..", "plots"))
filename  = joinpath(plots_dir, "tfim_correlator_vs_g_$(Libc.strftime("%Y-%m-%d_%Hh%M", time())).png")
savefig(p, filename)
println("Figure saved to $filename")
