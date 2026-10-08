# Example uses of the structure factor

using StructureFactorCertify
using Dualization, Mosek, MosekTools
using Printf

optimizer = Dualization.dual_optimizer(Mosek.Optimizer)

g = 0.5
system = Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, g))
exact  = tfim_nn_correlator_numerical(g)

## Nearest-neighbour correlator ⟨Z₀Z₁⟩
prob = StructureFactor_Problem(system, tf_single_correlator(1))
(lo, hi), model = compute_bounds(prob, Relaxation(L = 5, d = 2); optimizer)
@printf("%.7f ≤ ⟨Z₀Z₁⟩ ≤ %.7f   (exact %.7f)\n", lo, hi, exact)

## Commutator vs anticommutator KMS
for kms in (CommutatorKMS(), AnticommutatorKMS())
    t = @elapsed (lo, hi), _ = compute_bounds(prob, Relaxation(L = 5, d = 2, kms = kms); optimizer, verbose = false)
    @printf("  %-20s: [%.7f, %.7f]  %.1f s\n", nameof(typeof(kms)), lo, hi, t)
end

## Effect of the symmetry reduction (same bounds, smaller SDP; without symmetries L=5 already takes minutes)
for (use_time_reversal, use_parity) in ((true, true), (false, false))
    relax = Relaxation(L = 3, d = 2; use_time_reversal, use_parity)
    t = @elapsed (lo, hi), _ = compute_bounds(prob, relax; optimizer, verbose = false)
    @printf("  time reversal = %-5s, parity = %-5s: [%.7f, %.7f]  %.1f s\n", use_time_reversal, use_parity, lo, hi, t)
end

## General test function: Fejér kernel at p = 0 (probes ferromagnetic correlations)
tf = tf_fejer(2, 0.0, 1.0)
display(tf)
(lo, hi), _ = compute_bounds(StructureFactor_Problem(system, tf), Relaxation(L = 5, d = 2); optimizer)
