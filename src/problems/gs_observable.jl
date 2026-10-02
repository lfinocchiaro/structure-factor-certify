"""
    GS_Observable_Problem(system, observable)

Upper and lower bounds on the ground-state expectation value of `observable`, a PauliPolynomial
defined on the L sites of the relaxation.
"""
struct GS_Observable_Problem{T<:Spin_Lattice} <: Abstract_Problem
    system::T
    observable::PauliPolynomial # Note: to be generalized for non-qubits
end

"""
    compute_bounds(prob::GS_Observable_Problem, relax; optimizer, verbose = true) -> ((O_min, O_max), model)

Minimise and maximise ρ(observable) over the relaxation `relax` (with KMS), solved with `optimizer`.
"""
function compute_bounds(prob::GS_Observable_Problem{<:Spin_Lattice_1D}, relax::Relaxation; optimizer, verbose::Bool = true)
    verbose && println("── Ground state observable, $(prob.system.interaction), L=$(relax.L), d=$(relax.d) ──")
    return observable_bounds(prob.system, prob.observable, relax; optimizer, verbose)
end

# Lower and upper bounds on ρ(observable), shared with the structure factor problem
function observable_bounds(system::Spin_Lattice_1D, observable::PauliPolynomial, relax::Relaxation; optimizer, verbose::Bool)
    model = Model(optimizer)
    verbose || set_silent(model)
    obj = build_relaxation!(model, system, relax, observable; with_kms = true, verbose)

    O_min = optimize_objective!(model, MIN_SENSE, obj, "lower bound")
    O_max = optimize_objective!(model, MAX_SENSE, obj, "upper bound")
    verbose && println("  Certified bound: $O_min ≤ ⟨O⟩ ≤ $O_max")
    return (O_min, O_max), model
end
