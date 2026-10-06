## -----------
# Interactions
# ------------

abstract type Interaction end

"""
    interaction_range(int::Interaction) -> Int

Number of consecutive sites coupled by a single term of the Hamiltonian.
"""
interaction_range(int::Interaction) = error("Interaction range of $int has not been specified")

"Transverse-field Ising chain, H = -J ∑ Z_i Z_{i+1} + g ∑ X_i."
struct TFIM_1D_Interaction <: Interaction
    J::Float64
    g::Float64
end

interaction_range(::TFIM_1D_Interaction) = 2

"XY chain, H = J ∑ (1+γ) X_i X_{i+1} + (1-γ) Y_i Y_{i+1}."
struct XY_1D_Interaction <: Interaction
    J::Float64
    γ::Float64
end

interaction_range(::XY_1D_Interaction) = 2

"Heisenberg chain, H = (J/4) ∑ X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}."
struct Heisenberg_1D_Interaction <: Interaction
    J::Float64
end

interaction_range(::Heisenberg_1D_Interaction) = 2

"Transverse-field Ising model on the square lattice, H = -J ∑_⟨ij⟩ Z_i Z_j + g ∑ X_i."
struct TFIM_2D_Interaction <: Interaction
    J::Float64
    g::Float64
end

interaction_range(::TFIM_2D_Interaction) = 2

## -------
# Lattices
# --------

abstract type Spin_Lattice end

"""
    Spin_Lattice_1D(local_algebra_dimension, interaction)

Infinite 1D chain of spins with nearest-neighbour `interaction`. Only qubits (`local_algebra_dimension = 2`) are supported.
"""
struct Spin_Lattice_1D <: Spin_Lattice
    local_algebra_dimension::Int
    interaction::Interaction
    Spin_Lattice_1D(lad::Int, int::Interaction) = lad != 2 ? error("Not implemented yet for non-qubits") : new(lad, int)
end

"""
    Square_Lattice_2D(local_algebra_dimension, interaction)

Infinite square lattice of spins with nearest-neighbour `interaction`. Site (x, y) of an L×L window
has the linear index (y-1)L + x. Only qubits (`local_algebra_dimension = 2`) are supported.
"""
struct Square_Lattice_2D <: Spin_Lattice
    local_algebra_dimension::Int
    interaction::Interaction
    Square_Lattice_2D(lad::Int, int::Interaction) = lad != 2 ? error("Not implemented yet for non-qubits") : new(lad, int)
end

"""
    num_sites(system, L) -> Int

Number of sites of the window of linear size `L`: L for a chain, L² for a square lattice.
"""
num_sites(::Spin_Lattice_1D, L::Int) = L
num_sites(::Square_Lattice_2D, L::Int) = L^2


# -----
# Convert interactions into Hamiltonians (pauli polynomials)
# -----

# Polynomial ∑ c · term from a list of (c, term) pairs
function hamiltonian_from_terms(n::Int, terms::Vector{Tuple{ComplexF64,Vector{Int}}})
    coefs = Dict{Int,ComplexF64}()
    for (c, t) in terms
        coefs[unique_id(PauliMonomial(n, t))] = c
    end
    return PauliPolynomial(n, coefs)
end

"""
    local_term(n, i, ops...) -> Vector{Int}

Pauli string on `n` sites with letters `ops` on consecutive sites starting at site `i`, identity elsewhere.
"""
local_term(n::Int, i::Int, ops::Int...) = [zeros(Int, i-1); collect(ops); zeros(Int, n-i-length(ops)+1)]

# Pauli string with letters `a` on sites `i` and `j`, on a chain of n sites
_correlator(n::Int, i::Int, j::Int, a::Int) = local_term(n, i, a, zeros(Int, j-i-1)..., a)

"""
    get_hamiltonian(int::Interaction, n) -> PauliPolynomial

Hamiltonian of the interaction restricted to an open chain of `n` sites.
"""
get_hamiltonian(int::Interaction, n::Int) = error("get_hamiltonian() not implemented for Interaction $int")

function get_hamiltonian(int::TFIM_1D_Interaction, n::Int)
    terms = Tuple{ComplexF64,Vector{Int}}[]
    for i in 1:(n-1)
        push!(terms, (-int.J, local_term(n, i, 3, 3)))   # -J Z_i Z_{i+1}
    end
    for i in 1:n
        push!(terms, (int.g, local_term(n, i, 1)))       # g X_i
    end
    return hamiltonian_from_terms(n, terms)
end

function get_hamiltonian(int::XY_1D_Interaction, n::Int)
    terms = Tuple{ComplexF64,Vector{Int}}[]
    for i in 1:(n-1)
        push!(terms, (int.J*(1+int.γ), local_term(n, i, 1, 1)))   # J (1+γ) X_i X_{i+1}
        push!(terms, (int.J*(1-int.γ), local_term(n, i, 2, 2)))   # J (1-γ) Y_i Y_{i+1}
    end
    return hamiltonian_from_terms(n, terms)
end

function get_hamiltonian(int::Heisenberg_1D_Interaction, n::Int)
    terms = Tuple{ComplexF64,Vector{Int}}[]
    for i in 1:(n-1), a in 1:3
        push!(terms, (int.J/4, local_term(n, i, a, a)))           # (J/4) σ^a_i σ^a_{i+1}
    end
    return hamiltonian_from_terms(n, terms)
end



function get_hamiltonian(int::TFIM_2D_Interaction, n::Int)
    terms = Tuple{ComplexF64,Vector{Int}}[]
    L = isqrt(n)
    L^2 == n || error("n must be a perfect square for 2D lattice (n = $n)")
    for x in 1:L, y in 1:L
        i = (y-1)*L + x
        if x < L
            push!(terms, (-int.J, _correlator(n, i, i+1, 3)))   # -J Z_i Z_{i+1} (horizontal)
        end
        if y < L
            push!(terms, (-int.J, _correlator(n, i, i+L, 3)))   # -J Z_i Z_{i+L} (vertical)
        end
        push!(terms, (int.g, local_term(n, i, 1)))           # g X_i
    end
    return hamiltonian_from_terms(n, terms)
end
