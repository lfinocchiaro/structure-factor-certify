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

# Pauli string with letters `ops` on consecutive sites starting at `i`, on a chain of n sites
_local_term(n::Int, i::Int, ops::Int...) = [zeros(Int, i-1); collect(ops); zeros(Int, n-i-length(ops)+1)]

"""
    get_hamiltonian(int::Interaction, n) -> PauliPolynomial

Hamiltonian of the interaction restricted to an open chain of `n` sites.
"""
get_hamiltonian(int::Interaction, n::Int) = error("get_hamiltonian() not implemented for Interaction $int")

function get_hamiltonian(int::TFIM_1D_Interaction, n::Int)
    terms = Tuple{ComplexF64,Vector{Int}}[]
    for i in 1:(n-1)
        push!(terms, (-int.J, _local_term(n, i, 3, 3)))   # -J Z_i Z_{i+1}
    end
    for i in 1:n
        push!(terms, (int.g, _local_term(n, i, 1)))       # g X_i
    end
    return hamiltonian_from_terms(n, terms)
end

function get_hamiltonian(int::XY_1D_Interaction, n::Int)
    terms = Tuple{ComplexF64,Vector{Int}}[]
    for i in 1:(n-1)
        push!(terms, (int.J*(1+int.γ), _local_term(n, i, 1, 1)))   # J (1+γ) X_i X_{i+1}
        push!(terms, (int.J*(1-int.γ), _local_term(n, i, 2, 2)))   # J (1-γ) Y_i Y_{i+1}
    end
    return hamiltonian_from_terms(n, terms)
end

function get_hamiltonian(int::Heisenberg_1D_Interaction, n::Int)
    terms = Tuple{ComplexF64,Vector{Int}}[]
    for i in 1:(n-1), a in 1:3
        push!(terms, (int.J/4, _local_term(n, i, a, a)))           # (J/4) σ^a_i σ^a_{i+1}
    end
    return hamiltonian_from_terms(n, terms)
end
