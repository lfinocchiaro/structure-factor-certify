# Bounds on ground state expectation values with TFIM interactions

using StructureFactorCertify
using Dualization, Mosek, MosekTools
using Printf

optimizer = Dualization.dual_optimizer(Mosek.Optimizer)

## Magnetization ⟨Z₂⟩ on a 3-site chain
L = 3
prob = GS_Observable_Problem(Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, 0.75)),
                             from_monomial(PauliMonomial(L, [0, 3, 0])))
(lo, hi), _ = compute_bounds(prob, Relaxation(L = L, d = 2); optimizer)
println("$lo ≤ ⟨Z₂⟩ ≤ $hi")

## Correlator ⟨Z₂Z₃⟩ on a 4-site chain
L, g = 4, 0.8
prob = GS_Observable_Problem(Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, g)),
                             from_monomial(PauliMonomial(L, [0, 3, 3, 0])))
(lo, hi), _ = compute_bounds(prob, Relaxation(L = L, d = 2); optimizer)
println("$lo ≤ ⟨Z₂Z₃⟩ ≤ $hi   (exact: $(tfim_nn_correlator_numerical(g)))")

## Correlator ⟨Z₃Z₄⟩ on a 6-site chain: monomial basis and translation invariance
L, g = 6, 0.8
prob = GS_Observable_Problem(Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, g)),
                             from_monomial(PauliMonomial(L, [0, 0, 3, 3, 0, 0])))
println("exact: $(tfim_nn_correlator_numerical(g))")
for (basis, d) in (("full", 2), ("contiguous", 2), ("contiguous", 3)), use_translation in (false, true)
    relax = Relaxation(L = L, d = d; basis, use_translation)
    t = @elapsed (lo, hi), _ = compute_bounds(prob, relax; optimizer, verbose = false)
    @printf("  %-10s d=%d, translation = %-5s : [%.7f, %.7f]  %.1f s\n", basis, d, use_translation, lo, hi, t)
end
