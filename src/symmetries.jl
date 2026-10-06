## -------
# Symmetries
#
# Exact reductions of the SDP, detected from the Hamiltonian and the objective:
#   - Time reversal: if every term has an even number of Y's (real matrices), ρ may be taken real.
#     Then ρ(Q) = 0 when #Y(Q) is odd, and in the rescaled basis Bᵢ = i^{#Y(Pᵢ)} Pᵢ the PSD
#     matrices are real symmetric instead of complex Hermitian.
#   - Global Z₂ symmetries ∏X, ∏Y, ∏Z: ρ(Q) = 0 for odd Q, and the PSD matrices split into one
#     block per symmetry sector of the basis monomials.
# Justification: constraints and objective are invariant under ρ ↦ ρ∘σ, so averaging a feasible ρ
# over the symmetry group gives a symmetric feasible ρ with the same objective value.
# --------

"""
    Z2Symmetry(name, flipped)

Global Z₂ symmetry ∏σ, described by the two Pauli letters `flipped` that anticommute with σ.
"""
struct Z2Symmetry
    name::String
    flipped::Tuple{Int,Int}
end

# ∏X flips Y,Z   ∏Y flips X,Z   ∏Z flips X,Y
const Z2_CANDIDATES = (Z2Symmetry("∏X", (2, 3)), Z2Symmetry("∏Y", (1, 3)), Z2Symmetry("∏Z", (1, 2)))

"Number of Y letters in the Pauli string `term`."
count_Y(term) = count(==(2), term)

"Parity (0 or 1) of the Pauli string `term` under the Z₂ symmetry `sym`."
z2_parity(term, sym::Z2Symmetry) = count(in(sym.flipped), term) % 2

"""
    Symmetries(time_reversal, z2)

Symmetries used to reduce the SDP: `time_reversal::Bool` and a list `z2` of independent Z₂ generators.
"""
struct Symmetries
    time_reversal::Bool
    z2::Vector{Z2Symmetry}
end

function Base.show(io::IO, s::Symmetries)
    z2 = isempty(s.z2) ? "none" : join((sym.name for sym in s.z2), ", ")
    print(io, "time reversal = $(s.time_reversal), Z₂ = $z2")
end

"""
    detect_symmetries(polys::Vector{PauliPolynomial}) -> Symmetries

Symmetries shared by all `polys` (typically the Hamiltonian and the objective). Only independent Z₂
generators are kept: if ∏X, ∏Y, ∏Z are all symmetries, ∏Y = ∏X∏Z is dropped.
"""
function detect_symmetries(polys::Vector{PauliPolynomial})
    terms = [pauli_from_id(p.n, uid).term for p in polys for (uid, c) in p.coefs if abs(c) > ZERO_TOL]
    time_reversal = all(t -> iseven(count_Y(t)), terms)   # Hermitian polynomial is real iff no term has odd #Y
    z2 = Z2Symmetry[]
    for sym in Z2_CANDIDATES
        length(z2) == 2 && break
        all(t -> z2_parity(t, sym) == 0, terms) && push!(z2, sym)
    end
    return Symmetries(time_reversal, z2)
end


## -------
# Translation invariance
#
# For a translation-invariant state, ρ(Q) = ρ(τ_a Q) for every shift a (the state lives on the infinite
# lattice, so this holds even if τ_a Q leaves the window). Each Pauli string is therefore identified with
# its translate whose support starts at the origin of the window. Using it restricts the optimisation to
# translation-invariant states (equivalently: the objective is replaced by its translation average).
# --------

"""
    translate_to_origin(system, L, term) -> Vector{Int}

Translate of the Pauli string `term` (on the window of linear size `L`) whose support starts at site 1
(1D), or at the smallest x and smallest y of the window (2D).
"""
function translate_to_origin(::Spin_Lattice_1D, L::Int, term::Vector{Int})
    first = findfirst(!=(0), term)
    first === nothing && return term
    return [term[first:end]; zeros(Int, first-1)]
end

function translate_to_origin(::Square_Lattice_2D, L::Int, term::Vector{Int})
    occupied = findall(!=(0), term)
    isempty(occupied) && return term
    dx = minimum((s-1) % L for s in occupied)   # shift along x
    dy = minimum((s-1) ÷ L for s in occupied)   # shift along y
    shifted = zeros(Int, L^2)
    for s in occupied
        shifted[s - dy*L - dx] = term[s]
    end
    return shifted
end
