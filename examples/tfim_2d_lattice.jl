# Tests on 2D lattice with TFIM

using StructureFactorCertify
using SCS
using Dualization, Mosek, MosekTools
using Printf

optimizer_scs   = SCS.Optimizer   # Mosek crashes on the ⟨Z₁₁⟩ problem below with the full basis (single 352×352 block)
optimizer_mosek = Dualization.dual_optimizer(Mosek.Optimizer)

g = 0.5
system = Square_Lattice_2D(2, TFIM_2D_Interaction(1.0, g))

## Ground state energy on 2x2 lattice
# (no translation invariance here: the exact value is the ground energy of the isolated 2x2 plaquette)
L = 2
prob = GS_Energy_Problem(system)
E_min, _ = compute_bounds(prob, Relaxation(L = L, d = 2); optimizer = optimizer_scs, verbose = true)
println("E₀ ≥ $E_min   (exact: $(exact_ground_energy(get_hamiltonian(system.interaction, num_sites(system, L)))))")


## Magnetization ⟨ Z₁₁ ⟩ on a 3x3 lattice 0,1,2 x 0,1,2, with translation invariance

L = 3
obs_vect = local_term(L^2, (L÷2)*L + L÷2 + 1, 3) # Z₁₁
prob = GS_Observable_Problem(system, from_monomial(PauliMonomial(L^2, obs_vect)))

(lo, hi), _ = compute_bounds(prob, Relaxation(L = L, d = 2, use_translation = true); optimizer = optimizer_scs, verbose = true)
println("$lo ≤ ⟨Z⟩ ≤ $hi")


## Nearest-neighbour correlator ⟨ Z₁₁ Z₂₁ ⟩ on a 3x3 lattice (even under ∏X: two blocks, Mosek works)
# Basis sizes on the 3x3 window: "full" d=2 → 352 monomials, "contiguous" d=2 (all strings in 2x2 squares,
# up to 4-body plaquette terms) → 964 monomials

L = 3
obs_vect = local_term(L^2, (L÷2)*L + L÷2 + 1, 3, 3) # Z₁₁ Z₂₁
prob = GS_Observable_Problem(system, from_monomial(PauliMonomial(L^2, obs_vect)))


for (basis, d, use_translation) in (("full", 2, false), ("full", 2, true))#, ("contiguous", 2, true))
    relax = Relaxation(L = L, d = d; basis, use_translation)
    t = @elapsed (lo, hi), _ = compute_bounds(prob, relax; optimizer = optimizer_mosek, verbose = false)
    @printf("  %-10s d=%d, translation = %-5s : %.7f ≤ ⟨Z₁₁Z₂₁⟩ ≤ %.7f  %.1f s\n", basis, d, use_translation, lo, hi, t)
end
println("Careful, when we use translation invariance the meaning of ⟨Z₁₁Z₂₁⟩ is the average over all nearest-neighbour pairs, not just the one at (1,1)-(2,1).")
