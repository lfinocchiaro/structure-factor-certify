## -------
# Reduced density matrices
#
# For a set S of k sites, ρ_S = 2⁻ᵏ ∑_Q ρ(Q) Q ≽ 0, Q running over the 4ᵏ Pauli strings on S. Its entries
# are linear in the same moment variables as the moment matrix, so it is automatically consistent with it.
# The constraint is implied by M ≽ 0 when the basis contains every string on S, and is useful (and much
# cheaper than the equivalent 4ᵏ × 4ᵏ moment block) when S is larger than what the basis covers.
# --------

"""
    rdm_boxes(system, L, k, use_translation) -> Vector{Vector{Int}}

Sites of the reduced density matrices: every segment of `k` sites (1D) or k×k square (2D) of the window.
With translation invariance all of them give the same matrix, so only one is kept.
"""
function rdm_boxes(system::Spin_Lattice, L::Int, k::Int, use_translation::Bool)
    boxes = unique!(filter(box -> length(box) == num_sites(system, k), local_boxes(system, L, window_sites(system, L), k)))
    isempty(boxes) && error("rdm_size=$k does not fit in the window (L=$L)")
    return use_translation ? boxes[1:1] : boxes
end

"""
    add_rdm_constraint!(mm, box) -> Int

Constrain the reduced density matrix ρ_S on the sites `box` to be PSD (real symmetric if time reversal is
used, Hermitian otherwise). Returns its size 2ᵏ.
"""
function add_rdm_constraint!(mm::MomentMap, box::Vector{Int})
    k = length(box)
    dim = 2^k
    R_re = [zero(AffExpr) for _ in 1:dim, _ in 1:dim]   # real and imaginary parts of ρ_S
    R_im = [zero(AffExpr) for _ in 1:dim, _ in 1:dim]
    for letters in Iterators.product(fill(0:3, k)...)
        term = zeros(Int, mm.L)
        term[box] .= letters
        ρQ, _ = expectation(mm, from_monomial(PauliMonomial(mm.L, term)))   # ρ(Q) is real (Q Hermitian)
        is_zero_expr(ρQ) && continue
        Q = kron((PAULI_MATRICES[t+1] for t in reverse(collect(letters)))...)   # box[1] is the rightmost factor
        for idx in findall(!iszero, Q)
            add_to_expression!(R_re[idx], real(Q[idx]) / dim, ρQ)
            add_to_expression!(R_im[idx], imag(Q[idx]) / dim, ρQ)
        end
    end
    if mm.symmetries.time_reversal
        all(is_zero_expr, R_im) || error("Reduced density matrix on $box is not real: time-reversal reduction is invalid here")
        @constraint(mm.model, Symmetric(R_re) in PSDCone())
    else
        @constraint(mm.model, Hermitian(R_re .+ 1.0im .* R_im) in HermitianPSDCone())
    end
    return dim
end

"""
    add_rdm_constraints!(mm, system, L, k; use_translation) -> Vector{Int}

Add the PSD constraints of the reduced density matrices on `rdm_boxes(system, L, k, use_translation)`.
Returns their sizes.
"""
function add_rdm_constraints!(mm::MomentMap, system::Spin_Lattice, L::Int, k::Int; use_translation::Bool)
    return [add_rdm_constraint!(mm, box) for box in rdm_boxes(system, L, k, use_translation)]
end
