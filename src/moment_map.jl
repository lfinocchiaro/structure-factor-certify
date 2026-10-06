## -------
# Moment map: Pauli string → affine expression in the moment variables
#
# One real JuMP variable ρ(Q) per distinct Pauli string Q, created only when Q first appears and
# never for strings forced to 0 by symmetry. PSD matrices are affine expressions in these variables,
# so the algebraic equalities of the moment matrix (Mᵢᵢ = 1, Mᵢⱼ ∝ Mₖₗ when PᵢPⱼ ∝ PₖPₗ) are built in.
# --------

"""
    MomentMap(model, L, symmetries, canonical_uid = identity)

Lazy map from Pauli strings on `L` sites to real JuMP variables `moments[uid] = ρ(Q)` of `model`.
Strings with the same `canonical_uid` (e.g. translates of each other) share the same variable.
"""
struct MomentMap
    model::Model
    L::Int
    symmetries::Symmetries
    canonical_uid::Function           # uid → uid of the representative of its equivalence class
    moments::Dict{Int,VariableRef}   # canonical uid → ρ(Q), only for Q not forced to 0
end

MomentMap(model::Model, L::Int, symmetries::Symmetries, canonical_uid::Function = identity) =
    MomentMap(model, L, symmetries, canonical_uid, Dict{Int,VariableRef}())

"Tuple of Z₂ parities labelling the symmetry sector of the Pauli string `term`."
symmetry_sector(mm::MomentMap, term) = Tuple(z2_parity(term, sym) for sym in mm.symmetries.z2)

"Whether ρ(Q) = 0 is enforced by symmetry for the Pauli string `term`."
function vanishes_by_symmetry(mm::MomentMap, term)
    return (mm.symmetries.time_reversal && isodd(count_Y(term))) || any(sym -> z2_parity(term, sym) == 1, mm.symmetries.z2)
end

"JuMP variable ρ(Q) for the Pauli string with id `uid`, created on first call."
function moment_variable!(mm::MomentMap, uid::Int)
    return get!(mm.moments, uid) do
        @variable(mm.model, base_name = "y[$uid]")
    end
end

"""
    expectation(mm, poly, phase = 1) -> (re::AffExpr, im::AffExpr)

Real and imaginary parts of ρ(phase · poly), as affine expressions in the moment variables.
"""
function expectation(mm::MomentMap, poly::PauliPolynomial, phase::ComplexF64 = 1.0 + 0im)
    re, im_ = zero(AffExpr), zero(AffExpr)
    for (uid, c) in poly.coefs
        c *= phase
        abs(c) < ZERO_TOL && continue
        if uid == 0   # ρ(I) = 1
            add_to_expression!(re, real(c))
            add_to_expression!(im_, imag(c))
            continue
        end
        vanishes_by_symmetry(mm, pauli_from_id(mm.L, uid).term) && continue
        y = moment_variable!(mm, mm.canonical_uid(uid))
        add_to_expression!(re, real(c), y)
        add_to_expression!(im_, imag(c), y)
    end
    return re, im_
end

"Whether the affine expression `e` is (numerically) identically zero."
is_zero_expr(e::AffExpr) = abs(constant(e)) < ZERO_TOL && all(abs(c) < ZERO_TOL for c in values(e.terms))

# Phase of the real basis element B = i^{#Y} P  (1 when time reversal is not used)
real_basis_phase(mm::MomentMap, term) = mm.symmetries.time_reversal ? (1.0im)^count_Y(term) : 1.0 + 0im

"""
    add_psd_blocks!(mm, monomials, entry) -> Vector{Int}

Constrain the matrix X[i,j] = ρ(entry(i,j)), indexed by `monomials`, to be PSD, with one block per
symmetry sector. `entry(i,j)` returns a PauliPolynomial with entry(j,i) = entry(i,j)^*.
Returns the block sizes, largest first.
"""
function add_psd_blocks!(mm::MomentMap, monomials::Vector{PauliMonomial}, entry::Function)
    sectors = Dict{Any,Vector{Int}}()
    for (i, P) in enumerate(monomials)
        push!(get!(sectors, symmetry_sector(mm, P.term), Int[]), i)
    end
    time_reversal = mm.symmetries.time_reversal
    sizes = Int[]
    for idx in values(sectors)
        n = length(idx)
        push!(sizes, n)
        X = Matrix{time_reversal ? AffExpr : GenericAffExpr{ComplexF64,VariableRef}}(undef, n, n)
        for a in 1:n, b in a:n
            i, j = idx[a], idx[b]
            phase = conj(real_basis_phase(mm, monomials[i].term)) * real_basis_phase(mm, monomials[j].term)
            re, im_ = expectation(mm, entry(i, j), phase)
            if time_reversal
                is_zero_expr(im_) || error("Entry ($i,$j) is not real: time-reversal reduction is invalid here")
                X[a, b] = X[b, a] = re
            else
                X[a, b] = re + 1.0im * im_
                X[b, a] = re - 1.0im * im_
            end
        end
        if time_reversal
            @constraint(mm.model, Symmetric(X) in PSDCone())
        else
            @constraint(mm.model, Hermitian(X) in HermitianPSDCone())
        end
    end
    return sort(sizes; rev = true)
end
