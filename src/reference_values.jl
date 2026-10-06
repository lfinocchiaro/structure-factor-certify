## -------
# Analytical reference values (for testing)
# --------

"""
    dense_matrix(p::PauliPolynomial) -> Matrix{ComplexF64}

The 2ⁿ × 2ⁿ matrix of `p` (site 1 is the rightmost tensor factor). Only for small n.
"""
function dense_matrix(p::PauliPolynomial)
    M = zeros(ComplexF64, 2^p.n, 2^p.n)
    for (uid, c) in p.coefs
        M .+= c .* kron((PAULI_MATRICES[t+1] for t in reverse(pauli_from_id(p.n, uid).term))...)
    end
    return M
end

"""
    exact_ground_energy(H::PauliPolynomial) -> Float64

Lowest eigenvalue of `H`, by exact diagonalization of its dense matrix. Only for small n.
"""
exact_ground_energy(H::PauliPolynomial) = eigmin(Hermitian(dense_matrix(H)))

"""
    tfim_nn_correlator_numerical(g) -> Float64

Ground-state correlator ρ(Z_0 Z_1) of the infinite TFIM chain (J = 1, transverse field g),
  ρ(Z_0 Z_1) = 1/π ∫_0^π (1 + g cos k) / √(1 + 2g cos k + g²) dk,
computed with a midpoint rule (10⁴ points).
"""
function tfim_nn_correlator_numerical(g::Real)
    g == 0 && return 1.0
    n = 10_000
    dk = π / n
    s = 0.0
    for i in 0:(n-1)
        k = (i + 0.5) * dk
        s += (1 + g*cos(k)) / sqrt(1 + 2g*cos(k) + g^2)
    end
    return s * dk / π
end
