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
    Relaxation(; L, d, basis = "full", kms = CommutatorKMS(), rdm_size = 0, use_time_reversal = true,
                 use_parity = true, use_translation = false)

Level of the hierarchy: lattice truncated to a window of linear size `L` (L sites for a chain, L×L for a
square lattice), and monomial basis `basis`:
- `"full"`: all Pauli strings of degree ≤ `d` in the window,
- `"contiguous"`: all Pauli strings supported on `d` consecutive sites (1D) or in a d×d square (2D).
KMS condition `kms` (`CommutatorKMS()` or `AnticommutatorKMS()`). With `rdm_size = k > 0`, the reduced
density matrices on `k` consecutive sites (1D) or k×k squares (2D) are constrained to be PSD (2ᵏ×2ᵏ, resp.
2^{k²}×2^{k²} matrices). The exact symmetry reductions can be disabled for checking.
`use_translation = true` restricts to translation-invariant states.
"""
Base.@kwdef struct Relaxation
    L::Int
    d::Int
    basis::String = "full"
    kms::KMSCondition = CommutatorKMS()
    rdm_size::Int = 0
    use_time_reversal::Bool = true
    use_parity::Bool = true
    use_translation::Bool = false
end

abstract type Abstract_Problem end

compute_bounds(pb::Abstract_Problem, relax::Relaxation; kwargs...) = error("$pb not implemented yet")


"""
    window_sites(system, L; edge = true) -> Vector{Int}

Linear indices of the sites of the window of linear size `L`. With `edge = false`, only the sites whose
coordinates all lie in ℓ:(L+1-ℓ), with ℓ = interaction_range(system.interaction), so that every
Hamiltonian term touching them is inside the window.
"""
function window_sites(system::Spin_Lattice_1D, L::Int; edge::Bool = true)
    ℓ = interaction_range(system.interaction)
    return collect(edge ? (1:L) : (ℓ:(L+1-ℓ)))
end

function window_sites(system::Square_Lattice_2D, L::Int; edge::Bool = true)
    ℓ = interaction_range(system.interaction)
    r = edge ? (1:L) : (ℓ:(L+1-ℓ))
    return [(y-1)*L + x for y in r for x in r]
end

"""
    local_boxes(system, L, sites, d) -> Vector{Vector{Int}}

All segments of `d` consecutive sites (1D) or d×d squares (2D) of the window, intersected with `sites`.
"""
function local_boxes(::Spin_Lattice_1D, L::Int, sites::Vector{Int}, d::Int)
    return [filter(s -> x0 ≤ s < x0 + d, sites) for x0 in 1:L]
end

function local_boxes(::Square_Lattice_2D, L::Int, sites::Vector{Int}, d::Int)
    in_box(s, x0, y0) = x0 ≤ (s-1) % L + 1 < x0 + d && y0 ≤ (s-1) ÷ L + 1 < y0 + d
    return [filter(s -> in_box(s, x0, y0), sites) for y0 in 1:L for x0 in 1:L]
end

"""
    monomial_supports(system, L, sites, d, basis) -> Vector{Vector{Int}}

Supports (sorted lists of sites among `sites`) of the non-identity monomials of the basis:
all sets of ≤ `d` sites for `"full"`, all sets inside one of the `local_boxes` for `"contiguous"`.
"""
function monomial_supports(system::Spin_Lattice, L::Int, sites::Vector{Int}, d::Int, basis::String)
    if basis == "full"
        return [locs for k in 1:min(d, length(sites)) for locs in combinations(sites, k)]
    elseif basis == "contiguous"
        supports = Set{Vector{Int}}()
        for box in local_boxes(system, L, sites, d), k in 1:length(box), locs in combinations(box, k)
            push!(supports, locs)
        end
        return sort!(collect(supports); by = locs -> (length(locs), locs))
    else
        error("Unknown basis \"$basis\": use \"full\" or \"contiguous\"")
    end
end

"""
    build_monomial_basis(system, L, d; edge = true, basis = "full") -> Vector{PauliMonomial}

Pauli strings with supports `monomial_supports(system, L, window_sites(system, L; edge), d, basis)`,
identity first.
"""
function build_monomial_basis(system::Spin_Lattice, L::Int, d::Int; edge::Bool = true, basis::String = "full")
    system.local_algebra_dimension == 2 || error("Not implemented") # Assert qubits
    n = num_sites(system, L)
    monomials = [PauliMonomial(n, zeros(Int, n))]
    for locs in monomial_supports(system, L, window_sites(system, L; edge), d, basis), ops in Iterators.product(fill(1:3, length(locs))...)
        term = zeros(Int, n)
        term[locs] .= ops
        push!(monomials, PauliMonomial(n, term))
    end
    return monomials
end


"""
    build_relaxation!(model, system, relax, objective; with_kms, verbose) -> AffExpr

Add the moment variables and the PSD (and, if `with_kms`, KMS and stationarity) constraints of the
relaxation to `model`. Returns ρ(objective) as an affine expression.
"""
function build_relaxation!(model::Model, system::Spin_Lattice, relax::Relaxation, objective::PauliPolynomial;
                           with_kms::Bool, verbose::Bool)
    L, d = relax.L, relax.d
    n = num_sites(system, L)
    d ≥ 1 || error("d must be ≥ 1")
    objective.n == n || error("The objective is defined on $(objective.n) sites, but the relaxation window has $n sites (L=$L)")
    t0 = time()

    H = get_hamiltonian(system.interaction, n)
    detected = detect_symmetries([H, objective])
    symmetries = Symmetries(detected.time_reversal && relax.use_time_reversal, relax.use_parity ? detected.z2 : Z2Symmetry[])
    canonical_uid = relax.use_translation ?
        (uid -> unique_id(PauliMonomial(n, translate_to_origin(system, L, pauli_from_id(n, uid).term)))) : identity
    mm = MomentMap(model, n, symmetries, canonical_uid)

    # (P)+(A) moment matrix M[i,j] = ρ(Pᵢ Pⱼ)
    full = build_monomial_basis(system, L, d; edge = true, basis = relax.basis)
    sizes_M = add_psd_blocks!(mm, full, (i, j) -> begin
        phase, Q = full[i] * full[j]
        from_monomial(Q, phase)
    end)

    # (R) reduced density matrices ρ_S ≽ 0
    sizes_R = relax.rdm_size > 0 ? add_rdm_constraints!(mm, system, L, relax.rdm_size; relax.use_translation) : Int[]

    # (K) KMS matrix and stationarity
    sizes_N, n_stat = Int[], 0
    if with_kms
        ℓ = interaction_range(system.interaction)
        L ≥ 2ℓ-1 || error("L=$L too small for interaction range ℓ=$ℓ. Need L ≥ $(2ℓ-1).")
        inner = build_monomial_basis(system, L, d; edge = false, basis = relax.basis)
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
        println("  Symmetries used  : $symmetries, translation = $(relax.use_translation)")
        println("  Moment variables : $(length(mm.moments))")
        println("  M blocks         : $sizes_M  (full basis $(length(full)), $(symmetries.time_reversal ? "real" : "complex"))")
        isempty(sizes_R) || println("  RDM blocks       : $(length(sizes_R)) × $(first(sizes_R))")
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
