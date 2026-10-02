"""
    GS_Energy_Problem(system)

Lower bound on the ground-state energy ρ(H) of `system` truncated to L sites (positivity and Pauli
algebra only: the KMS condition of the relaxation is not used).
"""
struct GS_Energy_Problem{T<:Spin_Lattice} <: Abstract_Problem
    system::T
end

"""
    compute_bounds(prob::GS_Energy_Problem, relax; optimizer, verbose = true) -> (E_min, model)

Minimise ρ(H) over the relaxation `relax`, solved with `optimizer`.
"""
function compute_bounds(prob::GS_Energy_Problem{<:Spin_Lattice_1D}, relax::Relaxation; optimizer, verbose::Bool = true)
    verbose && println("── Ground state energy, $(prob.system.interaction), L=$(relax.L), d=$(relax.d) ──")

    model = Model(optimizer)
    verbose || set_silent(model)
    H = get_hamiltonian(prob.system.interaction, relax.L)
    obj = build_relaxation!(model, prob.system, relax, H; with_kms = false, verbose)

    E_min = optimize_objective!(model, MIN_SENSE, obj, "energy")
    verbose && println("  ⟨H⟩ ≥ $E_min")
    return E_min, model
end
