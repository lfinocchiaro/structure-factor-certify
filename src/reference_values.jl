## -------
# Analytical reference values (for testing)
# --------

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
