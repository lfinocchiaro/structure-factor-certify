# Lower bound on the ground state energy of a 2-site chain with TFIM interaction

using StructureFactorCertify
using Dualization, Mosek, MosekTools

optimizer = Dualization.dual_optimizer(Mosek.Optimizer)

g = 0.75
prob  = GS_Energy_Problem(Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, g)))
relax = Relaxation(L = 2, d = 2)

E_min, model = compute_bounds(prob, relax; optimizer)
println("⟨H⟩ ≥ $E_min   (exact minimal eigenvalue: $(-sqrt(1 + 4g^2)))")
