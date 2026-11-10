## -------
# Exact thermodynamic-limit correlations f(x) = ⟨Z_0 Z_x⟩_GS of free-fermion chains
# --------
#=
TFIM  H = -J Σ_x Z_x Z_{x+1} + g Σ_x X_x,  J > 0  (depends on g/J only)
  f(x) = Toeplitz determinant (Pfeuty 1970, Barouch–McCoy 1971)
  f̂(p) = m² δ(p) + f̂_reg(p),  m² = (1-(g/J)²)^{1/4} for |g| < J, 0 otherwise

XY    H = J Σ_x ((1+γ) X_x X_{x+1} + (1-γ) Y_x Y_{x+1}),  J ≠ 0  (Z-correlations depend on γ only)
  f(x) = -G_x G_{-x} for x ≠ 0 (Wick, Lieb–Schultz–Mattis 1961 §4, Barouch–McCoy 1971 at h = 0)
  G_r  = (1/2π) ∫ e^{-ikr} (cos k - iγ sin k)/Λ_k dk,  Λ_k = √(cos²k + γ² sin²k)
  f(x) = 0 for even x ≠ 0, f is even in γ, and f̂ has no Dirac part (gapped for γ ≠ 0, f ∼ 1/x² at γ = 0)

Pauli convention: Z² = 1, so f(0) = 1. For a test function φ ∈ S_N, the SDP target value is the finite sum
  (f̂, φ̂) = Σ_{|x|≤N} φ(x) f(x).
=#

"""c_n = (1/2π) ∫ e^{-ink} s(k) dk for n = -nmax:nmax, returned as a real vector indexed by n + nmax + 1
(trapezoid rule, spectrally accurate for smooth s). The symbols used here have real coefficients."""
function _fourier_coeffs(s, nmax; M = 2^16)
    k = 2π .* (0:M-1) ./ M
    sk = s.(k)
    a = zeros(2nmax + 1)
    a[nmax+1] = real(sum(sk)) / M
    zp, zm = copy(sk), copy(sk)
    em, ep = cis.(-k), cis.(k)
    for m in 1:nmax                       # build s·e^{∓imk} incrementally (avoids cis in the loop)
        zp .*= em; a[nmax+1+m] = real(sum(zp)) / M
        zm .*= ep; a[nmax+1-m] = real(sum(zm)) / M
    end
    return a
end

# TFIM: a_n = (1/2π) ∫ e^{-ink} (1 - g e^{ik})/|1 - g e^{ik}| dk, n = -nmax:nmax (g = field / J)
function _tfim_symbol_coeffs(g, nmax; M = 2^16)
    isapprox(g, 1.0) && return [2 / (π * (1 - 2m)) for m in -nmax:nmax]   # exact at criticality
    return _fourier_coeffs(k -> (1 - g * cis(k)) / abs(1 - g * cis(k)), nmax; M)
end

# XY: G_r for r = -rmax:rmax. At γ = 0 the symbol is sign(cos k) and G_r = 2 sin(πr/2)/(πr) in closed form.
# For small γ ≠ 0 the trapezoid error is ∼ exp(-γM), negligible for γ ≳ 1e-3 at the default M.
function _xy_G(γ, rmax; M = 2^16)
    iszero(γ) && return [r == 0 ? 0.0 : 2 * sin(π * r / 2) / (π * r) for r in -rmax:rmax]
    return _fourier_coeffs(k -> (cos(k) - im * γ * sin(k)) / sqrt(cos(k)^2 + γ^2 * sin(k)^2), rmax; M)
end

_tfim_ratio(int::TFIM_1D_Interaction) =
    int.J > 0 ? int.g / int.J : error("Exact TFIM correlations implemented for J > 0 only, got J = $(int.J)")

"""
    exact_correlations(interaction, R) -> Vector{Float64}
    exact_correlations(system::Spin_Lattice_1D, R)

Ground-state correlations f(r) = ⟨Z_0 Z_r⟩ of the infinite chain for r = 0:R, with `f[r+1] = f(r)`.
Implemented for `TFIM_1D_Interaction` (J > 0) and `XY_1D_Interaction` (J ≠ 0).
"""
exact_correlations(int::Interaction, R::Int) = error("exact_correlations not implemented for $int")
exact_correlations(system::Spin_Lattice_1D, R::Int) = exact_correlations(system.interaction, R)

function exact_correlations(int::TFIM_1D_Interaction, R::Int)
    a = _tfim_symbol_coeffs(abs(_tfim_ratio(int)), R)     # f depends on |g|: X → -X is a unitary symmetry
    T = [a[(i - j) + R + 1] for i in 1:R, j in 1:R]      # T_ij = a_{i-j}
    return [1.0; [det(T[1:r, 1:r]) for r in 1:R]]
end

function exact_correlations(int::XY_1D_Interaction, R::Int)
    iszero(int.J) && error("Exact XY correlations need J ≠ 0")
    G = _xy_G(int.γ, R)
    return [1.0; [-G[R+1+r] * G[R+1-r] for r in 1:R]]    # f(r) = ⟨Z⟩² - G_r G_{-r}, ⟨Z⟩ = 0
end

"""
    exact_dirac_weight(interaction) -> Float64
    exact_dirac_weight(system::Spin_Lattice_1D)

Weight m² = lim_{x→∞} f(x) of the Dirac peak of f̂ (at p = 0): (1 - (g/J)²)^{1/4} in the ordered phase
of the TFIM, 0 otherwise.
"""
exact_dirac_weight(system::Spin_Lattice_1D) = exact_dirac_weight(system.interaction)
exact_dirac_weight(int::TFIM_1D_Interaction) = (g = abs(_tfim_ratio(int)); g < 1 ? (1 - g^2)^(1/4) : 0.0)
exact_dirac_weight(::XY_1D_Interaction) = 0.0

"""
    exact_overlap(system, tf::TestFunction; f = exact_correlations(system, tf.N)) -> Float64

Exact value of (f̂, φ̂) = Σ_{|x|≤N} φ(x) f(x), the target of the SDP bounds of `StructureFactor_Problem(system, tf)`.
Pass precomputed correlations `f` (of length ≥ N+1) to reuse them for several test functions.
For `tf = tf_fejer(N, p)` this is the Fejér-smoothed structure factor (f̂ ⋆ K_N)(p), since f̂ is even.
"""
function exact_overlap(system::Spin_Lattice_1D, tf::TestFunction; f = exact_correlations(system, tf.N))
    length(f) ≥ tf.N + 1 || error("Need f(r) up to r = $(tf.N), got $(length(f) - 1)")
    return sum(phi_at(tf, x) * f[abs(x) + 1] for x in -tf.N:tf.N)
end

"""
    exact_structure_factor(system, p; R = 400, cesaro = false, f = exact_correlations(system, R)) -> Vector{Float64}

Regular part f̂_reg(p) = Σ_{|x|≤R} (f(x) - m²) e^{2iπpx} of the structure factor, for each p in `p`
(the Dirac part m² δ(p) is removed, see `exact_dirac_weight`). `cesaro = true` applies a Fejér window,
which removes Gibbs ringing when f decays slowly (e.g. at criticality).
"""
function exact_structure_factor(system::Spin_Lattice_1D, p; R::Int = 400, cesaro = false, f = exact_correlations(system, R))
    x = 0:R
    w = [r == 0 ? 1.0 : 2.0 for r in x]
    cesaro && (w .*= 1 .- x ./ (R + 1))
    fr = f[1:R+1] .- exact_dirac_weight(system)
    return [sum(w .* fr .* cos.(2π * pp .* x)) for pp in p]
end
