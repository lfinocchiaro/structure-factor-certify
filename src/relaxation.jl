## -------
# SDP relaxation shared by all problems
#
#   (P)+(A) moment matrix   M[i,j] = ρ(Pᵢ Pⱼ) ≽ 0,  Pᵢ in the full basis (all L sites)
#   (K)     KMS matrix      N[i,j] = ρ(KMS entry) ≽ 0,  Pᵢ in the inner basis (away from the edges)
#           + stationarity  ρ([H, ·]) = 0
# --------

## KMS conditions

abstract type KMSCondition end

"""
    CommutatorKMS()

KMS matrix N[i,j] = ρ(Pᵢ[H,Pⱼ] - [H,Pᵢ]Pⱼ), with single-monomial stationarity ρ([H,Pᵢ]) = 0.
"""
struct CommutatorKMS <: KMSCondition end

"""
    AnticommutatorKMS()

KMS matrix N[i,j] = ρ(PᵢHPⱼ - ½(PᵢPⱼH + HPᵢPⱼ)), with double-monomial stationarity ρ([H,PᵢPⱼ]) = 0.
"""
struct AnticommutatorKMS <: KMSCondition end

"""
    kms_entry_function(kms, H, P::Vector{PauliPolynomial}) -> (i, j) -> PauliPolynomial

Entries of the KMS matrix indexed by the inner basis `P`.
"""
function kms_entry_function(::CommutatorKMS, H::PauliPolynomial, P::Vector{PauliPolynomial})
    C = [commutator(H, Pᵢ) for Pᵢ in P]   # cached [H, Pᵢ]
    return (i, j) -> P[i] * C[j] - C[i] * P[j]
end

function kms_entry_function(::AnticommutatorKMS, H::PauliPolynomial, P::Vector{PauliPolynomial})
    HP = [H * Pᵢ for Pᵢ in P]   # cached H Pᵢ
    PH = [Pᵢ * H for Pᵢ in P]   # cached Pᵢ H
    return (i, j) -> PH[i] * P[j] - 0.5 * (P[i] * PH[j] + HP[i] * P[j])
end

"""
    stationarity_polynomials(kms, H, inner::Vector{PauliMonomial}) -> Vector{PauliPolynomial}

Polynomials A such that the relaxation imposes ρ(A) = 0.
"""
function stationarity_polynomials(::CommutatorKMS, H::PauliPolynomial, inner::Vector{PauliMonomial})
    return [commutator(H, from_monomial(P)) for P in inner]
end

function stationarity_polynomials(::AnticommutatorKMS, H::PauliPolynomial, inner::Vector{PauliMonomial})
    products = Dict{Int,PauliMonomial}()   # distinct PᵢPⱼ, up to a phase
    for i in eachindex(inner), j in i:lastindex(inner)
        _, Q = inner[i] * inner[j]
        products[unique_id(Q)] = Q
    end
    return [commutator(H, from_monomial(Q)) for Q in values(products)]
end


## Relaxation parameters

"""
    Relaxation(; L, d, kms = CommutatorKMS(), use_time_reversal = true, use_parity = true)

Level of the hierarchy: chain truncated to `L` sites, monomials of degree ≤ `d`, KMS condition `kms`
(`CommutatorKMS()` or `AnticommutatorKMS()`). The symmetry reductions can be disabled for checking.
"""
Base.@kwdef struct Relaxation
    L::Int
    d::Int
    kms::KMSCondition = CommutatorKMS()
    use_time_reversal::Bool = true
    use_parity::Bool = true
end

abstract type Abstract_Problem end

compute_bounds(pb::Abstract_Problem, relax::Relaxation; kwargs...) = error("$pb not implemented yet")


"""
    build_monomial_basis(system, L, d; edge = true) -> Vector{PauliMonomial}

Pauli strings of degree ≤ `d` on a chain of `L` sites (identity first). With `edge = false`, only
sites ℓ:(L+1-ℓ) are used, where ℓ = interaction_range(system.interaction).
"""
function build_monomial_basis(system::Spin_Lattice_1D, L::Int, d::Int; edge::Bool = true)
    system.local_algebra_dimension == 2 || error("Not implemented") # Assert qubits
    ℓ = interaction_range(system.interaction)
    sites = edge ? (1:L) : (ℓ:(L+1-ℓ))
    basis = [PauliMonomial(L, zeros(Int, L))]
    for k in 1:d, locs in combinations(sites, k), ops in Iterators.product(fill(1:3, k)...)
        term = zeros(Int, L)
        term[locs] .= ops
        push!(basis, PauliMonomial(L, term))
    end
    return basis
end


"""
    build_relaxation!(model, system, relax, objective; with_kms, verbose) -> AffExpr

Add the moment variables and the PSD (and, if `with_kms`, KMS and stationarity) constraints of the
relaxation to `model`. Returns ρ(objective) as an affine expression.
"""
function build_relaxation!(model::Model, system::Spin_Lattice_1D, relax::Relaxation, objective::PauliPolynomial;
                           with_kms::Bool, verbose::Bool)
    L, d = relax.L, relax.d
    d ≥ 1 || error("d must be ≥ 1")
    objective.n == L || error("The objective is defined on $(objective.n) sites, but the relaxation uses L=$L")
    t0 = time()

    H = get_hamiltonian(system.interaction, L)
    detected = detect_symmetries([H, objective])
    symmetries = Symmetries(detected.time_reversal && relax.use_time_reversal, relax.use_parity ? detected.z2 : Z2Symmetry[])
    mm = MomentMap(model, L, symmetries)

    # (P)+(A) moment matrix M[i,j] = ρ(Pᵢ Pⱼ)
    full = build_monomial_basis(system, L, d; edge = true)
    sizes_M = add_psd_blocks!(mm, full, (i, j) -> begin
        phase, Q = full[i] * full[j]
        from_monomial(Q, phase)
    end)

    # (K) KMS matrix and stationarity
    sizes_N, n_stat = Int[], 0
    if with_kms
        ℓ = interaction_range(system.interaction)
        L ≥ 2ℓ-1 || error("L=$L too small for interaction range ℓ=$ℓ. Need L ≥ $(2ℓ-1).")
        inner = build_monomial_basis(system, L, d; edge = false)
        sizes_N = add_psd_blocks!(mm, inner, kms_entry_function(relax.kms, H, from_monomial.(inner)))
        for A in stationarity_polynomials(relax.kms, H, inner), e in expectation(mm, A)
            is_zero_expr(e) && continue
            @constraint(model, e == 0)
            n_stat += 1
        end
    end

    n_moments = length(mm.moments)
    obj, obj_im = expectation(mm, objective)
    length(mm.moments) == n_moments || error("The objective contains Pauli strings that do not appear in the relaxation (increase d or L)")
    is_zero_expr(obj_im) || @warn "Objective has a non-zero imaginary part; using its real part"

    if verbose
        println("  Symmetries used  : $symmetries")
        println("  Moment variables : $(length(mm.moments))")
        println("  M blocks         : $sizes_M  (full basis $(length(full)), $(symmetries.time_reversal ? "real" : "complex"))")
        if with_kms
            println("  N blocks         : $sizes_N  ($(nameof(typeof(relax.kms))))")
            println("  Stationarity eqs : $n_stat")
        end
        @printf("  Model built in %.2f s\n", time() - t0)
    end
    return obj
end


"Warn if the last solve of `model` did not terminate properly."
function check_status(model::Model, label::String)
    st = termination_status(model)
    st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.SLOW_PROGRESS) || @warn "Solver status for $label: $st (result may be unreliable)"
end

"""
    optimize_objective!(model, sense, obj, label) -> Float64

Optimise `obj` in direction `sense` (MIN_SENSE or MAX_SENSE) and return the optimal value.
"""
function optimize_objective!(model::Model, sense::MOI.OptimizationSense, obj::AffExpr, label::String)
    @objective(model, sense, obj)
    optimize!(model)
    check_status(model, label)
    return objective_value(model)
end
